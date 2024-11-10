import json

class TreeNode:
    def __init__(self, file_path=None, class_names=None, package_import_paths=None, package=None, imports=None, import_objects = None, functions=None, property_declarations=None, exports=None, summary = None):
        self.file_path = file_path
        self.class_names = class_names or []
        self.package_import_paths = package_import_paths or {}
        self.package = package or []
        self.imports = imports or []
        self.import_objects = import_objects or {}
        self.exports = exports or []
        self.property_declarations = property_declarations or []
        self.functions = functions or []
        self.summary = summary or ''

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "class_names": self.class_names,
            "imports": self.imports,
            "import_objects":self.import_objects,
            "exports": self.exports,
            "package_import_paths": self.package_import_paths,
            "package": self.package,
            "property_declarations": self.property_declarations,
            "functions": [func.to_dict() for func in self.functions],
            "summary": self.summary
        }

    # def __repr__(self):
    #     functions = "\n\n".join([str(func) for func in self.functions])
    #     return (
    #         f"TreeNode:\nFile Path:{self.file_path}\nClass Names: {self.class_names}\n"
    #         f"Imports: {self.imports}\nExports: {self.exports}\nProperties: {self.property_declarations}\n"
    #         f"Functions:\n{functions}\nPackage Paths:{self.package_import_paths}\nPackage: {self.package}"
    #     )
    
    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)