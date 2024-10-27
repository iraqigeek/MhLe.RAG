import re
from .FunctionNode_module import FunctionNode

def traverse_tree_go(node, code, node_tree, language):
    go_function = None
    query_string = """
    (import_declaration) @import
    (package_clause) @package
    (function_declaration) @function
    (method_declaration) @method
    (type_declaration) @type
    (var_declaration) @var
    """
    query = language.query(query_string)
    captures = query.captures(node)

    should_traverse_children = True
    for capture_node, capture_name in captures:
        if capture_node == node:
            should_traverse_children = False

    for capture_node, capture_index in captures:
        text = code[capture_node.start_byte : capture_node.end_byte].decode("utf-8").strip()

        if capture_index == "import":
            node_tree.imports.append(text)
        elif capture_index == "package":
            package_name_match = re.search(r'package\s+(\w+)', text)
            if package_name_match:
                node_tree.package = package_name_match.group(1)
        elif capture_index in ["function", "method"]:
            function_details = extract_function_details_go(text)
            if function_details and not any(f.name == function_details.name for f in node_tree.functions):
                node_tree.functions.append(function_details)
        elif capture_index == "type":
            type_name_match = re.search(r'type\s+(\w+)\s+struct', text)
            if type_name_match:
                node_tree.class_names.append(type_name_match.group(1))
        elif capture_index == "var":
            node_tree.property_declarations.append(text)

    if should_traverse_children:
        for child in node.children:
            traverse_tree_go(child, code, node_tree, language)

def extract_function_details_go(text):
    func_name_match = re.search(r'func\s+(\w+)\s*\(', text)
    if not func_name_match:  # Handle methods
        func_name_match = re.search(r'func\s*\(\s*\w+\s+\*\w+\s*\)\s+(\w+)\s*\(', text)
    func_name = func_name_match.group(1) if func_name_match else "anonymous"
    parameters_match = re.search(r'\((.*?)\)', text)
    parameters = parameters_match.group(1).strip() if parameters_match else ""
    func_body_match = re.search(r'\{(.*)\}', text, re.DOTALL)
    func_body = func_body_match.group(1).strip() if func_body_match else ""

    return FunctionNode(
        name=func_name,
        parameters=parameters.split(",") if parameters else [],
        return_type="undefined",  # Go functions might not declare return types explicitly in all cases
        body=func_body
    )