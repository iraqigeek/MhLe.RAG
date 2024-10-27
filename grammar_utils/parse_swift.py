import re
from .FunctionNode_module import FunctionNode

def traverse_tree_swift(node, code, node_tree, language):
    query_string = """
    (import_declaration) @import
    (class_declaration) @class
    (function_declaration) @function
    (property_declaration) @variable
    """
    query = language.query(query_string)
    captures = query.captures(node)

    for capture_node, capture_name in captures:
        text = code[capture_node.start_byte : capture_node.end_byte].decode("utf-8").strip()

        if capture_name == "import":
            node_tree.imports.append(text)
        elif capture_name == "class":
            class_name_match = re.search(r'(class|struct|actor|extension|enum)\s+(\w+)', text)
            if class_name_match:
                node_tree.class_names.append(class_name_match.group(2))
        elif capture_name == "function":
            function_details = extract_function_details_swift(text)
            if function_details and not any(f.name == function_details.name for f in node_tree.functions):
                node_tree.functions.append(function_details)
        elif capture_name == "variable":
            node_tree.property_declarations.append(text)

    for child in node.children:
        traverse_tree_swift(child, code, node_tree, language)

def extract_function_details_swift(text):
    func_name_match = re.search(r'func\s+(\w+)\s*\(', text)
    func_name = func_name_match.group(1) if func_name_match else "anonymous"
    parameters_match = re.search(r'\((.*?)\)', text)
    parameters = parameters_match.group(1).strip() if parameters_match else ""
    return_type_match = re.search(r'->\s*(\w+)', text)
    return_type = return_type_match.group(1).strip() if return_type_match else "Void"
    func_body_match = re.search(r'\{(.*)\}', text, re.DOTALL)
    func_body = func_body_match.group(1).strip() if func_body_match else ""

    return FunctionNode(
        name=func_name,
        parameters=parameters.split(",") if parameters else [],
        return_type=return_type,
        body=func_body
    )
