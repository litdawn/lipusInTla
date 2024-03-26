import tla
from type import Type


class VarDefVisitor():
    def __init__(self):
        self.varnames = []

    def visit_Decl(self, node):
        if 'main' in node.name:
            return
        self.varnames.append(node.name)


def get_varnames_from_source_code(file):
    varnames = []
    tla_lines = []
    with open(file, 'r') as f:
        for line in f.readlines():
            tla_lines.append(line)
            if line.startswith("VARIABLE"):
                varnames.append(line[9:-1])
    return varnames, tla_lines


# "ResponseMatched(VARI,VARP)" 中的 ResponseMatched
def get_state_from_source_code(file):
    state = []
    return state


class ConstDefVisitor():
    def __init__(self):
        self.consts = set()

    def visit_Constant(self, node):
        if node.type == 'int':
            self.consts.add(int(node.value))


def get_consts_from_source_code(file):
    consters = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.startswith("CONSTANT"):
                consters.append(line[9:-1])
    return consters


# 主函数，输入一个json对象
def parse_file(file):
    content = file.body
    # tla_ins = tla.TLA()
    # 处理常量
    constants = []
    variables = []
    actions = []
    for param in content.declaredParams:
        constants.append({"name": param.paramName, "info": parse_type(param.typeComment)})
    for var in content.definedviables:
        variables.append({"name": var.variableName, "info": parse_type(var.typeComment)})
    for operator in content.operatorDefinitions:
        if operator.type == "action":
            actions.append({"name": operator.operatorName, "info": parse_type(operator.typeComment)})
    tla_ins = tla.TLA(constants, variables, actions)
    return tla_ins


#
def parse_type(str):
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
            "param_type": param_info.self_type,
            "param_num": param_info.num,
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
            "index_type": index_info.self_type,
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
    return {
        "name": "default",
        "self_type": Type.BOOL
    }
