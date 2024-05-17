import json
import os.path


class SeedTemplate:
    def __init__(self,tla_ins, path2configDotJson=""):
        json_data = {}
        try:
            with open(path2configDotJson, 'r') as f:
                self.json_data = json.load(f)
                json_data = self.json_data

        except FileNotFoundError:
            print("Error: failed to load the json file.")
            exit(1)
        self.preds = json_data["preds"]
        self.preds_alt = json_data["preds_alt"]
        self.safety = json_data["safety"]
        self.constants = json_data["constants"]
        self.constraints = json_data["constraint"]
        self.quant_inv = json_data["quant_inv"]
        self.quant_inv_alt = json_data["quant_inv_alt"]
        self.quant_vars = json_data["quant_vars"]
        self.model_consts = json_data["model_consts"]
        self.symmetry = json_data["symmetry"]
        self.typeok = json_data["typeok"]
        self.simulate = json_data["simulate"]
        self.tla_ins = tla_ins
        self.seeds = self.preds
        self.variables = json_data["variable"].split(",")

    def generate(self):
        return self.preds, self.quant_vars

    def get_quants(self):
        return self.quant_vars

    def get_seeds(self):
        return self.preds

    def get_constants(self):
        return self.constants
