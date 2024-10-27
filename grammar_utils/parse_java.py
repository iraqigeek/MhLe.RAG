import re
from .FunctionNode_module import FunctionNode

def traverse_tree_java(node, code, node_tree, language):
    java_function = None
    query_string = """
    (import_declaration) @import
    (package_declaration) @package
    (class_declaration name: (identifier) @name) @class
    (annotation) @annotation
    (interface_declaration name: (identifier) @name) @interface
    (field_declaration) @field
    (method_declaration) @method
    """

    query = language.query(query_string)
    captures = query.captures(node)

    should_traverse_children = True
    for capture_node, capture_name in captures:
        if capture_node == node:
            should_traverse_children = False

    for capture_node, capture_index in captures:
        if capture_index == "import":
            node_tree.imports.append(
                code[capture_node.start_byte : capture_node.end_byte]
                .decode("utf-8")
                .strip()
            )

        elif capture_index == "package":
            node_tree.package = (
                code[capture_node.start_byte : capture_node.end_byte]
                .decode("utf-8")
                .strip()
            )

        elif capture_index in ["class", "class_public", "class_abstract"]:
            class_name_match = re.search(
                r"\b(?:class|interface)\s+([a-zA-Z_]\w*)",
                code[capture_node.start_byte : capture_node.end_byte].decode("utf-8"),
            )
            if class_name_match:
                class_name = class_name_match.group(1)
                node_tree.class_names.append(class_name)

        elif capture_index == "field":
            property_declaration = (
                code[capture_node.start_byte : capture_node.end_byte]
                .decode("utf-8")
                .strip()
            )
            node_tree.property_declarations.append(property_declaration)

        elif capture_index == "annotation":
            annotation_text = (
                code[capture_node.start_byte : capture_node.end_byte]
                .decode("utf-8")
                .strip()
            )
            if java_function:
                java_function.annotations.append(annotation_text)

        elif capture_index == "method":
            method_code = code[capture_node.start_byte : capture_node.end_byte].decode(
                "utf-8"
            )
            func_name_match = re.search(
                r"\b(?:public|protected|private|static|final|abstract|synchronized|native|strictfp)?\s*(\w+\s+)?(\w+)\s*\(",
                method_code,
            )
            if func_name_match:
                return_type = func_name_match.group(1).strip() if func_name_match.group(1) else "void"
                func_name = func_name_match.group(2).strip()
                parameters_match = re.search(r'\((.*?)\)', method_code)
                parameters = parameters_match.group(1).strip() if parameters_match else ""
                func_body_match = re.search(r'\{(.*)\}', method_code, re.DOTALL)
                func_body = func_body_match.group(1).strip() if func_body_match else ""

                java_function = FunctionNode(
                    name=func_name,
                    parameters=parameters.split(",") if parameters else [],
                    return_type=return_type,
                    body=func_body,
                    class_names=node_tree.class_names,
                )

                duplicate_found = any(
                    func.name == java_function.name
                    and func.return_type == java_function.return_type
                    and func.parameters == java_function.parameters
                    for func in node_tree.functions
                )

                if not duplicate_found:
                    node_tree.functions.append(java_function)

    if should_traverse_children:
        for child_node in node.children:
            traverse_tree_java(child_node, code, node_tree, language)

