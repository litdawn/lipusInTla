from type import Type


class Element(object):
    def __init__(self):
        self.self_type = 0  # array, set, action # bool, string
        self.name = ""
        # set 有
        self.sub_num = 0
        self.sub_type = ""

        # action 有 todo 改成{}
        self.param_type = ""
        self.param_num = 0
        self.result = None

        # array有
        ''' 
        \* @ type: str -> set(str) node
        self_type: array 
        name: node
        content_type: set
        index_type: str
        '''
        self.content = None
        self.index_type = ""


# todo
class TLA:
    def __init__(self, constants, variables, actions):
        self.variables = variables
        self.actions = actions
        self.constants = constants


    def construct_var(self, name, info):
        var = self.Variable()
        var.self_type = info.self_type
        var.name = name
        if var.self_type == Type.ARRAY:
            var.index_type = info.index_type
            var.result = self.construct_var("result", info.result)

        elif var.self_type == Type.SET:
            var.sub_num = info.sub_num
            var.sub_type = info.sub_type
        return var

    class Variable(Element):
        #
        # \* @type: Set( << Str, Str, Str >>);
        def __init__(self):
            super(Element, self).__init__()

    class Constant(Element):
        sub = {}
        def __init__(self):
            super(Element, self).__init__()

    class Action(Element):
        def __init__(self, ):
            super(Element, self).__init__()
