import json

class ImportNode:
    def __init__(self, name, type, file_path, alias = None, docstring = None):
        self.name = name
        self.type = type
        self.file_path = file_path or None
        self.alias = alias or None
        self.docstring = docstring or None
        
    def to_dict(self):
        return {
            "name": self.name,
            "type": self.type,
            "file_path": self.file_path,
            "alias": self.alias,
            "docstring": self.docstring,
        }
    
    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

