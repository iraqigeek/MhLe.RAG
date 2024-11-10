import sys
import os
import re
import inspect
import importlib.util
import sysconfig
from .FunctionNode_module import FunctionNode
from .ImportNode_module import ImportNode
from .TreeNode_module import TreeNode

def find_python_package_directories(root_dir: str):
    # Helper function to check if a directory is a virtual environment
    def is_virtualenv(path):
        return os.path.isfile(os.path.join(path, 'pyvenv.cfg'))

    # Check for common virtual environment directories in the app root
    possible_envs = ['venv', '.venv', 'env', '.env']
    venv_site_packages = None

    for env_name in possible_envs:
        env_path = os.path.join(root_dir, env_name)
        if is_virtualenv(env_path):
            # Site-packages location varies by OS
            if os.name == 'nt':  # Windows
                venv_site_packages = os.path.join(env_path, 'Lib', 'site-packages')
            else:  # POSIX (Linux, MacOS)
                venv_site_packages = os.path.join(env_path, 'lib', f'python{sys.version[:3]}', 'site-packages')
            if os.path.isdir(venv_site_packages):
                break
            else:
                venv_site_packages = None  # Reset if not found

    # Get system libraries directory
    system_lib_dir = os.path.dirname(os.__file__)

    return venv_site_packages, system_lib_dir

def get_python_source_file(module_name, root_dir):
    venv_site_packages, system_lib_dir = find_python_package_directories(root_dir)
    
    stdlib_path = sysconfig.get_paths()['stdlib']
    if os.path.exists(stdlib_path + '/' + module_name):
            if os.path.exists(stdlib_path + '/' + module_name + '/__init__.py'):
                return stdlib_path + '/' + module_name + '/__init__.py'
    
    # Check If the module name is installed in the app's venv
    module_parts = module_name.split(".")
    module_path = os.path.join(venv_site_packages, *module_parts)
    if os.path.isdir(module_path) and os.path.exists(os.path.join(module_path, "__init__.py")):  # Package directory
        return os.path.join(module_path, "__init__.py")
    elif os.path.exists(venv_site_packages + '/' + "/".join(module_parts) + '.py'):
        return venv_site_packages + '/' + "/".join(module_parts) + '.py'
    
    # Check If the module name is a system library
    module_path = os.path.join(system_lib_dir, *module_parts)
    if os.path.isdir(module_path) and os.path.exists(os.path.join(module_path, "__init__.py")):  # Package directory
        return os.path.join(module_path, "__init__.py")
    elif os.path.exists(system_lib_dir + '/' + "/".join(module_parts) + '.py'):
        return system_lib_dir + '/' + "/".join(module_parts) + '.py'

    return None  # If no source file is found

def find_python_import(module_name, module_name_alias, import_name, import_name_alias, node_tree, parser, language, root_dir):
    #First check it's not a built-in module
    if module_name in sys.builtin_module_names:
        return None
    
    # Add module package to node_tree if not present
    if module_name not in node_tree.import_objects:
        file_name = get_python_source_file(module_name, root_dir)
        if not file_name:
            return None  # Module not found or cannot be located

        try:
            with open(file_name, 'r', encoding='utf-8') as file:
                source_code = file.read()
        except Exception as e:
            print("An error occurred:", e)

        module_tree = parser.parse(bytes(source_code, "utf8"))
        module_query = language.query("(module . (expression_statement (string) @docstring))")
        module_captures = module_query.captures(module_tree.root_node)
        if not module_captures:
            module_query = language.query("(module (expression_statement (string) @docstring))")
            module_captures = module_query.captures(module_tree.root_node)
        # if not module_captures:
        #    breakpoint()
        node_tree.import_objects[module_name] = ImportNode(module_name, "module", file_name, module_name_alias, ''.join([n.text.decode('utf-8') for n, na in module_captures]))

    if import_name:
        # Look for the specified import_name
        import_full_name = f"{module_name}.{import_name}"
        if import_full_name not in node_tree.import_objects: 
            # Load the source filer
            module = node_tree.import_objects[module_name]
            file_name = module.file_path
            with open(file_name, 'r', encoding='utf-8') as file:
                source_code = file.read()
            module_tree = parser.parse(bytes(source_code, "utf8"))
            # Check function definitions
            function_query = language.query("(function_definition) @function")
            function_captures = function_query.captures(module_tree.root_node)
            matches = [
                n
                for n, _ in function_captures
                for p in n.children
                if p.type == 'identifier' and p.text.decode('utf-8') == import_name
            ]
            if matches:
                docstring_query = language.query("(function_definition body: (block . (expression_statement (string) @docstring)))")
                docstring_captures = docstring_query.captures(matches[0])
                node_tree.import_objects[import_full_name] = ImportNode(import_full_name, "function", file_name, import_name_alias, ''.join([n.text.decode('utf-8') for n, na in docstring_captures]))
                return import_full_name
            
            # Check class definitions if not found in functions
            class_query = language.query("(class_definition body: (block . (expression_statement (string) @docstring)))")
            class_root = [n for n in module_tree.root_node.children 
                        for cn in n.children 
                        if n.type =='class_definition' 
                        if cn.type =='identifier' and cn.text.decode('utf-8') == import_name]
            if not class_root:
                class_root = [cn for n in module_tree.root_node.children 
                            for cn in n.children 
                            for gcn in cn.children 
                            if n.type =='decorated_definition' 
                            if cn.type =='class_definition' 
                            if gcn.type == 'identifier']
            if class_root:
                class_captures = class_query.captures(class_root[0])
                matches = [
                    n.text.decode('utf-8')
                    for n, _ in class_captures
                    for p in n.parent.parent.parent.children
                    if p.type == 'identifier' and p.text.decode('utf-8') == import_name
                ]

                docstring = ''.join([n.text.decode('utf-8') for n, na in class_captures])
                node_tree.import_objects[import_full_name] = ImportNode(import_full_name, "class", file_name, module_name_alias, docstring)
                return import_full_name

        # If not found, check recursively in the module's imports
        import_query = language.query("(import_statement) @import (import_from_statement) @import_from")
        import_captures = import_query.captures(module_tree.root_node)
        matches = [(n.children[1].text.decode('utf-8'), p.text.decode('utf-8'))  
                for n, na in import_captures 
                for p in n.children[2:] 
                if p.text.decode('utf-8')== import_name
                ]
        if len(matches) > 0:
            for module, imp in matches:
                if module.startswith('.'):
                    module = module_name + module
                result = find_python_import(module, None, imp, None, node_tree, parser, language, root_dir)
                if result:
                    return result
        for import_node, _ in import_captures:
            imported_module = [n.text.decode('utf-8') for n in import_node.children if n.type == 'dotted_name']
            if imported_module:
                imported_module = imported_module[0] 
            else:
                imported_module = [c.text.decode('utf-8') 
                                   for n in import_node.children 
                                   for c in n.children 
                                   if (n.type == 'dotted_name' or n.type == 'aliased_import') 
                                   and c.type == 'dotted_name'
                                   ]
                if len(imported_module) > 0:
                    imported_module= imported_module[0]
                else: 
                    breakpoint()
            result = find_python_import(imported_module, None, import_name, None, node_tree, parser, language, root_dir)
            if result:
                return result

        # Return None if nothing is found
        return None

def traverse_tree_python(node, code, node_tree, language, parser, root_dir):
    query_string = """
    (import_from_statement) @import_from
    (import_statement) @import
    (class_definition) @class
    (function_definition) @function
    (assignment) @variable
    (decorator) @decorator
    """
    query = language.query(query_string)
    captures = query.captures(node)

    #ALI: revisit later if needed
    '''
    should_traverse_children = True
    for capture_node, capture_name in captures:
        if capture_node == node:
            should_traverse_children = False
    '''

    for node, index in captures:
        extracted_text = code.strip()

        if index == "import":
            import_alias = [i.text.decode('utf-8') 
                            for n in node.children 
                            for i in n.children 
                            if n.type == 'aliased_import' 
                            and i.type == 'identifier'
                            ]
            module_name = [i.text.decode('utf-8') 
                           for n in node.children 
                           for c in n.children 
                           for i in c.children 
                           if n.type == 'aliased_import' 
                           and c.type == 'dotted_name' 
                           and i.type == 'identifier'
                           ]
            if not module_name and not import_alias:
                module_name = [n.text.decode('utf-8') 
                               for n in node.children  
                               if n.type ==  'dotted_name'
                               ]

            module_name = module_name[0]
            import_alias = import_alias[0] if len(import_alias) == 1 else import_alias

            if module_name:
                find_python_import(module_name, import_alias, None, None, node_tree, parser, language, root_dir)
            node_tree.imports.append(module_name)

        elif index == "import_from":
            parts = [child.text.decode('utf-8') for child in node.children if child.type == 'dotted_name']
            module_name = parts[0]
            if len(parts) > 1:
                for import_name in parts[1:]:
                    result = find_python_import(module_name, None, import_name, None, node_tree, parser, language, root_dir)
                    if module_name != '' and import_name != '':
                        node_tree.imports.append(f"from {module_name} import {import_name}")
                    else:
                        node_tree.imports.append(extracted_text)
            else:
                aliased_imports = [child
                                   for child in node.children 
                                   if child.grammar_name == 'aliased_import' 
                                   ]
                for alias in aliased_imports:
                    import_name = [child.text.decode('utf-8') for child in alias.children if child.type == 'dotted_name'][0]
                    alias_name = [child.text.decode('utf-8') for child in alias.children if child.type == 'identifier'][0]
                    result = find_python_import(module_name, None, import_name, alias_name, node_tree, parser, language, root_dir)
        elif index == "class":
            class_name_match = re.search(r'class\s+(\w+)', extracted_text)
            if class_name_match:
                node_tree.class_names.append(class_name_match.group(1))
        elif index == "function":
            function_details = extract_function_details_python(node, language)
            if function_details and not any(f.name == function_details.name for f in node_tree.functions):
                if node.parent is not None and node.parent.parent is not None and node.parent.parent.type =='class_definition':
                    function_details.class_name = node.parent.parent.child_by_field_name('name').text.decode('utf-8')
                node_tree.functions.append(function_details)
                
        elif index == "variable":
            if not node_tree.functions and not node_tree.class_names:
                node_tree.property_declarations.append(extracted_text)
    '''
    if should_traverse_children:
        for child in node.children:
            traverse_tree_python(child, code, node_tree, language)
    '''

def extract_function_details_python(node, language):
    # skip inner functions
    if node.parent.type == 'block' and node.parent.parent.type == 'function_definition':
        return
    body_node = [n for n in node.children if n.type == 'block'][0]
    function_calls = [c.text.decode('utf8') 
                      for n,_ in language.query("(call) @call").captures(body_node) 
                      for c in n.children 
                      if c.type =='identifier' or c.type == 'attribute'
                      ]
    function_calls = list(set(function_calls))
    body_text = body_node.text.decode('utf8')
    func_name = [n.text.decode('utf-8') 
                 for n in node.children 
                 if n.type == 'identifier'][0
                                            ]
    func_params = [p.text.decode('utf-8') 
                   for n in node.children 
                   for p in n.children 
                   if n.type == 'parameters' and p.type =='identifier'
                   ]
    
    decorators = []
    if node.parent.type == 'decorated_definition':
        decorators = [n.text.decode('utf-8') 
                      for d in node.parent.children 
                      for n in d.children
                      if d.type == 'decorator'
                      and n.type =='identifier'
                    ]

    return FunctionNode(
        name=func_name,
        parameters=func_params,
        function_calls=function_calls,
        annotations=decorators,
        return_type="None",
        body=body_text
    )

    return None