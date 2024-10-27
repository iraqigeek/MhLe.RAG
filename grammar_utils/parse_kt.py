import re
from .FunctionNode_module import FunctionNode

def append_to_package_import_paths(package, name, node_tree):
    package_import_path = f"{package}.{name}".strip("package ")
    node_tree.package_import_paths[package_import_path] = package_import_path

def extract_property_name(property_declaration):
    pattern = (
        r"\b(?:(?:public|private|protected)\s+)*(?:(?:static|final)\s+)*[a-zA-Z_]"
        r"\w*(?:<.*>)?(?:\[\])?\s+([a-zA-Z_]\w*)|(?:(?:val|var))\s+([a-zA-Z_]\w*)"
    )

    match = re.search(pattern, property_declaration)
    if match:
        java_prop_name = match.group(1)
        kotlin_prop_name = match.group(2)
        return java_prop_name or kotlin_prop_name

    return None


def traverse_tree_kt(node, code, node_tree, language):
    kotlin_function = None
    query_string = """
    (import_list) @import
    (package_header) @package
    (class_declaration) @class_or_interface
    (annotation) @annotation
    (object_declaration) @object_declaration
    (property_declaration) @field
    (function_declaration) @function
    """

    query = language.query(query_string)
    captures = query.captures(node)

    should_traverse_children = True
    for capture_node, capture_name in captures:
        if capture_node == node:
            should_traverse_children = False

    is_data_class = False
    for capture_node, capture_index in captures:
        if capture_index == "import":
            node_tree.imports = (
                code[capture_node.start_byte : capture_node.end_byte]
                .decode("utf-8")
                .strip()
            ).split("\n")

        elif capture_index == "package":
            node_tree.package = (
                code[capture_node.start_byte : capture_node.end_byte]
                .decode("utf-8")
                .strip()
            )

        elif capture_index == "class_or_interface":
            class_code = code[capture_node.start_byte : capture_node.end_byte].decode(
                "utf-8"
            )
            class_name_match = re.search(
                r"\b(?:sealed\s+class|data\s+class|class|interface)\s+([a-zA-Z_]\w*)",
                class_code,
            )
            if class_name_match:
                class_name = class_name_match.group(1)
                is_data_class = "data class" in class_code
                node_tree.class_names.append(f"{class_name}")
                node_tree.is_interface = "interface" in class_code

                # Extract data class fields
                if is_data_class:
                    # Modified regular expression to capture the entire line for each property
                    fields = re.findall(
                        r"\b(val|var)\s+([a-zA-Z_]\w*\s*:\s*[a-zA-Z_]\w*(\??)(<.*>)?(\??))",
                        class_code,
                    )
                    node_tree.property_declarations = (
                        ",\n".join(" ".join(f) for f in fields)
                    ).split("\n")

        elif capture_index == "annotation":
            annotation_text = (
                code[capture_node.start_byte : capture_node.end_byte]
                .decode("utf-8")
                .strip()
            )
            if kotlin_function:
                kotlin_function.annotations.append(annotation_text)

        # Added extraction of object declarations
        elif capture_index == "object_declaration":
            object_name_match = re.search(
                r"\b(?:object)\s+([a-zA-Z_]\w*)",
                code[capture_node.start_byte : capture_node.end_byte].decode("utf-8"),
            )
            if object_name_match:
                object_name = object_name_match.group(1)
                node_tree.class_names.append(object_name)

        elif capture_index == "field" and not is_data_class:
            property_declaration = (
                code[capture_node.start_byte : capture_node.end_byte]
                .decode("utf-8")
                .strip()
            )
            node_tree.property_declarations.append(property_declaration)

        elif capture_index == "function":
            function_code = code[
                capture_node.start_byte : capture_node.end_byte
            ].decode("utf-8")
            func_name_match = re.search(
                r"\b(?:fun)\s+(?:[a-zA-Z_]\w*\.)*([a-zA-Z_]\w*)", function_code
            )
            if func_name_match:
                func_name = func_name_match.group(1)
                parameters_match = re.search(r"\((.*?)\)", function_code)
                parameters = parameters_match.group(1) if parameters_match else ""
                return_type_match = re.search(
                    r":\s*([a-zA-Z_][\w<>,.? ]*)", function_code
                )
                return_type = (
                    return_type_match.group(1).strip() if return_type_match else "Unit"
                )
                func_body_match = re.search(r"\{(.*)\}", function_code, re.DOTALL)
                func_body = func_body_match.group(1).strip() if func_body_match else ""
                kotlin_function = FunctionNode(
                    func_name,
                    parameters.split(","),
                    return_type,
                    func_body,
                    class_names=node_tree.class_names,
                )

                # Check for duplicates
                duplicate_found = any(
                    func.name == kotlin_function.name
                    and func.return_type == kotlin_function.return_type
                    and func.parameters == kotlin_function.parameters
                    for func in node_tree.functions
                )
                if not duplicate_found:
                    node_tree.functions.append(kotlin_function)
    if should_traverse_children:
        for child in node.children:
            traverse_tree_kt(child, code, node_tree, language)
    else:
        # Append class names, function names, and property names to package_import_paths
        if node_tree.package:
            package_name = node_tree.package.replace(";", "").strip()
            if node_tree.class_names:
                for class_name in node_tree.class_names:
                    append_to_package_import_paths(package_name, class_name, node_tree)
            if node_tree.functions:
                for function in node_tree.functions:
                    append_to_package_import_paths(
                        package_name, function.name, node_tree
                    )
            if node_tree.property_declarations:
                for property_declaration in node_tree.property_declarations:
                    property_name = extract_property_name(property_declaration)
                    if property_name:
                        append_to_package_import_paths(
                            package_name, property_name, node_tree
                        )
