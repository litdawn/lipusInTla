import json
import os.path


class SeedTemplate:
    def __init__(self, path2configDotJson):
        json_data = ""
        try:
            with open(path2configDotJson, 'r') as f:
                json_data = json.load(f)
        except FileNotFoundError:
            print("Error: failed to load the json file.")
            exit(1)
        self.preds = json_data["preds"]
        self.preds_alt = json_data["preds_alt"]
        self.safety = json_data["safety"]
        self.constants = json_data["constants"]
        self.constants = json_data["constants"]
        self.quant_inv = json_data["quant_inv"]
        self.quant_inv_alt = json_data["quant_inv_alt"]
        self.quant_vars = json_data["quant_vars"]
        self.model_consts = json_data["model_consts"]
        self.symmetry = json_data["symmetry"]
        self.typeok = json_data["typeok"]
        self.simulate = json_data["simulate"]

    def generate(self):
        return self.preds, self.quant_vars