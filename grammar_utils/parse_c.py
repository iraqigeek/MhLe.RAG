import re
from .FunctionNode_module import FunctionNode

def traverse_tree_c(node, code, node_tree, language):
    c_function = None
    query_string = """
    (preproc_include) @include
    (function_definition) @function
    (declaration) @variable
    (struct_specifier) @struct
    """
    query = language.query(query_string)
    captures = query.captures(node)

    should_traverse_children = True
    for capture_node, capture_name in captures:
        if capture_node == node:
            should_traverse_children = False

    for capture_node, capture_index in captures:
        text = code[capture_node.start_byte : capture_node.end_byte].decode("utf-8").strip()

        if capture_index == "include":
            node_tree.imports.append(text)
        elif capture_index == "function":
            function_details = extract_function_details_c(text)
            if function_details and not any(f.name == function_details.name for f in node_tree.functions):
                node_tree.functions.append(function_details)
        elif capture_index == "variable":
            # Consider global variables only if outside any function definition
            if not node_tree.functions:
                node_tree.property_declarations.append(text)
        elif capture_index == "struct":
            struct_name_match = re.search(r'struct\s+(\w+)', text)
            if struct_name_match:
                node_tree.class_names.append(struct_name_match.group(1))

    if should_traverse_children:
        for child in node.children:
            traverse_tree_c(child, code, node_tree, language)

def extract_function_details_c(text):
    func_name_match = re.search(r'(\w+)\s*\(', text)
    func_name = func_name_match.group(1) if func_name_match else "anonymous"
    parameters_match = re.search(r'\((.*?)\)', text)
    parameters = parameters_match.group(1).strip() if parameters_match else ""
    return_type_match = re.search(r'^(\w+)\s+', text)
    return_type = return_type_match.group(1).strip() if return_type_match else "int"  # Default return type in C is int
    func_body_match = re.search(r'\{\s*(.*?)\s*\}', text, re.DOTALL)
    func_body = func_body_match.group(1).strip() if func_body_match else ""

    return FunctionNode(
        name=func_name,
        parameters=parameters.split(",") if parameters else [],
        return_type=return_type,
        body=func_body
    )