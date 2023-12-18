
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
    for extrac in [0, 1, -1, 2, -2, 3, -3, 6, 4]:
        if extrac not in consters:
            consters.append(extrac)
    return consters