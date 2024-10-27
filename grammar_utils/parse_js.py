import re
from .FunctionNode_module import FunctionNode

def traverse_tree_js(node, code, node_tree, language):
    js_function = None
    query_string = """
    (import_statement) @import
    (class_declaration) @class
    (function_declaration) @function
    (arrow_function) @arrow_function
    (method_definition) @method
    (variable_declarator) @variable
    (export_statement) @export
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
        elif capture_index == "class":
            class_name_match = re.search(r'class\s+(\w+)', text)
            if class_name_match:
                node_tree.class_names.append(class_name_match.group(1))
        elif capture_index in ["function", "arrow_function", "method"]:
            function_details = extract_function_details_js(text)
            if function_details and not any(f.name == function_details.name for f in node_tree.functions):
                node_tree.functions.append(function_details)
        elif capture_index == "variable":
            node_tree.property_declarations.append(text)
        elif capture_index == "export":
            node_tree.exports.append(text)  # Assuming you might want to track exports similarly

    if should_traverse_children:
        for child in node.children:
            traverse_tree_js(child, code, node_tree, language)

def extract_function_details_js(text):
    func_name_match = re.search(r'function\s+(\w+)\s*\(', text)
    if not func_name_match:  # Check for arrow functions or anonymous functions
        func_name_match = re.search(r'(\w+)\s*=\s*\(', text)
    func_name = func_name_match.group(1) if func_name_match else "anonymous"
    parameters_match = re.search(r'\((.*?)\)', text)
    parameters = parameters_match.group(1).strip() if parameters_match else ""
    func_body_match = re.search(r'\{(.*)\}', text, re.DOTALL)
    func_body = func_body_match.group(1).strip() if func_body_match else ""

    return FunctionNode(
        name=func_name,
        parameters=parameters.split(",") if parameters else [],
        return_type="n/a",  # JavaScript functions do not explicitly declare return types
        body=func_body
    )
