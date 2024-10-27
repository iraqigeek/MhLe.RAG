import json

class FunctionNode:
    def __init__(self, name, parameters, return_type, body, function_calls = None, is_abstract=False, class_names=None, annotations=None, summary=None):
        self.name = name
        self.parameters = parameters or []
        self.return_type = return_type
        self.body = body
        self.function_calls = function_calls or []
        self.is_abstract = is_abstract
        self.class_name = " ".join(class_names) if class_names else ""
        self.annotations = annotations or [],
        self.summary = summary or ''

    def to_dict(self):
        return {
            "name": self.name,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "body": self.body,
            "function_calls": self.function_calls,
            "is_abstract": self.is_abstract,
            "class_name": self.class_name,
            "annotations": self.annotations,
            "summary": self.summary
        }

    # def __repr__(self):
    #     parameter_str = ", ".join(self.parameters)
    #     return (
    #         f"\n\n------ Name: {self.name}\n------ Parameters: {parameter_str}\n------ Return Type: "
    #         f"{self.return_type}\n------ Body:\t{self.body}"
    #         f"\n------ Abstract:\t{self.is_abstract}\n"
    #         f"\n------ Annotations:\t{self.annotations}\n------ Class Name:\t{self.class_name}"
    #         f"\n------ Summary:\t{self.summary}"
    #     )

    def to_json(self):
        if isinstance(self.body, bytes):
            self.body = self.body.decode("utf-8")
        return json.dumps(self.__dict__, indent=4)