from seedTemplate.tlaParser.tla import TLA
from seedTemplate.SeedTemplate import Type
import json


def main(path2cfg, path2json):
    from seedTemplate.SeedTemplate import SeedTemplate
    with open(path2json) as json_file:
        json_data = json.load(json_file)
        with open(path2cfg) as cfg_file:
            cfg_data = cfg_file.read().split("\n")
            tla_ins = parse_file(json_data, cfg_data)
            seed_tmpl = SeedTemplate(tla_ins)
            seed_tmpl.generate()
            return tla_ins, seed_tmpl


# 主函数，输入一个json对象
def parse_file(json_data, cfg_data):
    content = json_data['body']
    # tla_ins = tla.TLA()
    # 处理常量
    constants = []
    variables = []
    actions = []
    states = []
    for param in content['declaredParams']:
        constants.append({"name": param['paramName'], "info": parse_type(param['typeComment'])})
    for var in content['definedVariables']:
        variables.append({"name": var['variableName'], "info": parse_type(var['typeComment'])})
    for operator in content['operatorDefinitions']:
        if operator['type'] == "Action":
            actions.append({"name": operator['operatorName'],
                            "info": parse_type(operator['typeComment'], operator['concreteContent'])})
        if operator['type'] == "State":
            states.append(
                {"name": operator['operatorName'], "info": parse_type(concrete_content=operator['concreteContent'])})

    tla_ins = TLA()
    tla_ins.init_var(constants, variables, actions, states)

    for line in cfg_data:
        if line.startswith("INIT"):
            tla_ins.init = line.split(" ")[-1]
        elif line.startswith("NEXT"):
            tla_ins.next = line.split(" ")[-1]
        elif line.startswith("INVARIANT"):
            tla_ins.inv = line.split(" ")[-1]
        elif line.startswith("CONSTANT"):
            if line.find("=") != 0:
                name = line.split("=")[0]
                value = line.split("=")[1]
                tla_ins.constants[name.replace("CONSTANT", "").strip()].real_val = value[1:-1].split(",")
                tla_ins.model_const = line
        elif line.startswith("\\*"):
            continue
        elif line.startswith("SYMMETRY"):
            tla_ins.sym = line.split(" ")[-1]
        elif line.startswith("CONSTRAINT"):
            tla_ins.constraint = line.split(" ")[-1]
        else:
            line = line.strip()
            if line.find("=") != 0:
                name = line.split("=")[0]
                value = line.split("=")[1]
                tla_ins.constants[name.strip()].real_val = value[1:-1].split(",")
    return tla_ins


#
def parse_type(str="", concrete_content=""):
    # @type: (Str, Str, Str, Str, Str) = > Bool
    # str = "Set(<<Str, Str, Str, Str>>)"

    # 一个action
    index = str.find("=>")
    if index != -1:
        param_part = str[0:index - 1].strip()
        param_info = parse_type(param_part)
        result_part = str[index + 2:].strip()
        result_info = parse_type(result_part)
        return {
            "self_type": Type.ACTION,
            "concrete_content": concrete_content,
            "param_type": param_info["self_type"],
            "param_num": param_info["num"],
            "result": result_info
        }

    # 一个数组
    index = str.find("->")
    if index != -1:
        index_part = str[0:index - 1].strip()
        index_info = parse_type(index_part)
        content_part = str[index + 2:].strip()
        content_info = parse_type(content_part)
        return {
            "self_type": Type.ARRAY,
            "index_type": index_info["self_type"],
            "content": content_info
        }

    # 一个集合
    index = str.find("Set")
    if index != -1:
        str_index = str.count("Str")
        bool_index = str.count("Bool")
        if str_index == 0:
            return {
                "self_type": Type.SET,
                "sub_num": bool_index,
                "sub_type": Type.BOOL
            }
        else:
            return {
                "self_type": Type.SET,
                "sub_num": str_index,
                "sub_type": Type.STRING
            }

    # 参数
    index = str.find(",")
    if index != -1:
        str_index = str.count("Str")
        bool_index = str.count("Bool")
        if str_index == 0:
            return {
                "self_type": Type.BOOL,
                "num": bool_index,
            }
        else:
            return {
                "self_type": Type.STRING,
                "num": str_index,
            }

    # 字符串
    index = str.find("Str")
    if index != -1:
        return {
            "name": "default",
            "self_type": Type.STRING
        }

    # Bool
    index = str.find("Bool")
    if index != -1:
        return {
            "name": "default",
            "self_type": Type.BOOL
        }

    # state
    if str == "":
        return {
            "self_type": Type.State,
            "concrete_content": concrete_content
        }

#
# if __name__ == "__main__":
#     main()
