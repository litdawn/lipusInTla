from seedTemplate.SeedTemplate import Type
from pyparsing import Forward, Combine, infixNotation, opAssoc, Keyword, Word, alphanums, Suppress, Optional, ZeroOrMore


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
        self.concrete_content = ""

        # state
        self.real = ""

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
    def __init__(self):
        self.variables = {}
        self.actions = {}
        self.states = {}
        self.constants = {}
        self.init = ""
        self.next = ""
        self.inv = ""

    def init_var(self, constants, variables, actions, states):
        for var in variables:
            self.variables.update({var["name"]: self.construct_var(var["name"], var["info"])})
        for act in actions:
            self.actions.update({act["name"]: self.construct_var(act["name"], act["info"])})
        for cons in constants:
            self.constants.update({cons["name"]: self.construct_var(cons["name"], cons["info"])})
        for state in states:
            self.states.update({state["name"]: self.construct_var(state["name"], state["info"])})

    def construct_var(self, name, info):
        var = self.Variable()
        var.self_type = info["self_type"]
        var.name = name
        if var.self_type == Type.ACTION:
            var.index_type = info["param_type"]
            var.param_num = info["param_num"]
            var.result = self.construct_var("result", info["result"])
        elif var.self_type == Type.SET:
            var.sub_num = info["sub_num"]
            var.sub_type = info["sub_type"]
        elif var.self_type == Type.ARRAY:
            var.index_type = info["index_type"]
            var.content = self.construct_var("content", info["content"])
        return var

    def duplicate_var(self, name, info):
        var = self.Variable()
        var.self_type = info.self_type
        var.name = name
        if var.self_type == Type.SET:
            var.sub_num = info.sub_num
            var.sub_type = info.sub_type
        elif var.self_type == Type.ARRAY:
            var.index_type = info.index_type
            var.content = info.content
        return var

    # 解析逻辑表达式, 结果形如
    # [
    #   ['ResponseMatched(VARI,VARP)', '\\/', '~',
    #       ['<<VARI,VARP>>', '\\in', 'response_sent']
    #   ]
    # ]
    @staticmethod
    def parse_logic_expression(expression):

        # 定义操作数、函数名和符号
        identifier = Word(alphanums + "_")
        function_name = Word(alphanums + "_")
        keyword_not = Keyword("~")
        keyword_subset = Keyword("\\subseteq")
        keyword_belongs_to = Keyword("\\in")
        keyword_and = Keyword("/\\")
        keyword_or = Keyword("\\/")

        # 定义括号
        LPAREN = Suppress("(")
        RPAREN = Suppress(")")

        # 定义操作符优先级
        precedence = [
            (keyword_subset, 2, opAssoc.LEFT),
            (keyword_belongs_to, 2, opAssoc.LEFT),
            (keyword_and, 2, opAssoc.LEFT),
            (keyword_or, 2, opAssoc.LEFT),
        ]

        # 定义逻辑表达式
        expr = Forward()
        atom = Forward()
        identifiers = Combine(
            "<<" + Optional(keyword_not) + identifier + ZeroOrMore("," + Optional(keyword_not) + identifier) + ">>")

        # 定义函数参数列表
        arg = (identifier | identifiers) | function_name | "~" + identifier
        args_list = arg + ZeroOrMore("," + arg)

        # 定义函数调用表达式
        function_call = Combine(function_name + "(" + Optional(args_list) + ")")
        atom <<= "~" + LPAREN + expr + RPAREN | LPAREN + expr + RPAREN | function_call | identifiers | identifier
        expr <<= infixNotation(atom, precedence)

        return  expr.parseString(expression)

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
