
class VarDefVisitor():
    def __init__(self):
        self.varnames = []

    def visit_Decl(self, node):
        if 'main' in node.name:
            return
        self.varnames.append(node.name)


def get_varnames_from_source_code(lines):
    varnames = []
    for line in lines:
        if line.startswith("VARIABLE"):
            varnames.append(line.substring(9))
    return varnames


class ConstDefVisitor():
    def __init__(self):
        self.consts = set()

    def visit_Constant(self, node):
        if node.type == 'int':
            self.consts.add(int(node.value))


def get_consts_from_source_code(lines):
    consters = []
    for line in lines:
        if line.startswith("CONSTANT"):
            consters.append(line.substring(9))

    #todo 这段是干嘛的
    for extrac in [0, 1, -1, 2, -2, 3, -3, 6, 4]:
        if extrac not in consters:
            consters.append(extrac)
    return consters
