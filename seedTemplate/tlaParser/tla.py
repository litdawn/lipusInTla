from typing import Dict

from seedTemplate.tlaParser.type import Type
from pyparsing import Forward, Combine, infixNotation, opAssoc, Keyword, Word, alphanums, Suppress, Optional, ZeroOrMore
import re
from PT_generators.RL_Prunning.Conifg import config


class Element(object):
    def __init__(self):
        self.self_type: Type = Type.DEFAULT  # array, set, action # bool, string
        self.name = ""
        # set 有
        self.sub_num = 0
        self.sub_type: Type = Type.DEFAULT

        # action 有 todo 改成{}
        self.param_type = ""
        self.param_num = 0
        self.result = None
        self.concrete_content = ""

        # state
        self.real = ""
        # 和self.concrete_content = ""

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


class TLA:
    def __init__(self):
        self.variables = {}
        self.actions = {}
        self.states = {}
        self.constants = {}
        self.init = ""
        self.next = ""
        self.inv = ""
        self.type_ok = "TypeOK"
        self.model_const = None

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
            var.concrete_content = info["concrete_content"]
            var.index_type = info["param_type"]
            var.param_num = info["param_num"]
            var.result = self.construct_var("result", info["result"])
        elif var.self_type == Type.SET:
            var.sub_num = info["sub_num"]
            var.sub_type = info["sub_type"]
        elif var.self_type == Type.ARRAY:
            var.index_type = info["index_type"]
            var.content = self.construct_var("content", info["content"])
        elif var.self_type == Type.STATE:
            var.index_type = 0
            var.concrete_content = info["concrete_content"]
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

        quant_reg = re.compile(r"\\[AE](.*?):")
        expression = expression.split("==")[-1].strip()[2:]
        expression = quant_reg.sub("", expression)

        if len(expression) == 0:
            return []

        # 定义操作数、函数名和符号
        identifier = Word(alphanums + "_" + "[]" + "{}")
        action_name = Word(alphanums + "_")
        keyword_not = Keyword("~")
        keyword_subset = Keyword("\\subseteq")
        keyword_belongs_to = Keyword("\\in")
        keyword_equal = Keyword("=")
        keyword_and = Keyword("/\\")
        keyword_or = Keyword("\\/")
        keyword_x = Keyword("\\X")
        keyword_to = Keyword("->")
        keyword_is = Keyword("|->")

        # 定义括号
        LPAREN = Suppress("(")
        RPAREN = Suppress(")")

        precedence = [
            (keyword_x, 2, opAssoc.LEFT),
            (keyword_subset, 2, opAssoc.LEFT),
            (keyword_is, 2, opAssoc.LEFT),
            (keyword_to, 2, opAssoc.LEFT),
            (keyword_belongs_to, 2, opAssoc.LEFT),
            (keyword_equal, 2, opAssoc.LEFT),
            (keyword_and, 2, opAssoc.LEFT),
            (keyword_or, 2, opAssoc.LEFT),
        ]

        expr = Forward()
        atom = Forward()
        identifiers = Combine(
            "<<" + Optional(keyword_not) + identifier + ZeroOrMore("," + Optional(keyword_not) + identifier) + ">>")

        arg = (identifier | identifiers) | action_name | "~" + identifier
        args_list = arg + ZeroOrMore("," + arg)

        action_call = Combine(action_name + "(" + Optional(args_list) + ")")

        atom <<= "~" + LPAREN + expr + RPAREN | LPAREN + expr + RPAREN | action_call | identifiers | identifier
        expr <<= infixNotation(atom, precedence)

        return expr.parseString(expression)

    class Variable(Element):
        #
        # \* @type: Set( << Str, Str, Str >>);
        def __init__(self):
            super(Element, self).__init__()

    class Constant(Element):

        def __init__(self):
            super(Element, self).__init__()

    class Action(Element):
        def __init__(self, ):
            super(Element, self).__init__()

# if __name__ == "__main__":
#     # quant_reg = re.compile(r"\\[AE](.*?):")
#     inputs = (
#                  "Init == "
#                  " /\\ semaphore = [i \\in Server |-> TRUE]"
#                  "/\\ clientlocks = [i \\in Client |-> {}]"
#              ).split("==")[-1].strip()[2:]
#
#     # inputs = quant_reg.sub("", inputs)
#     # print(inputs)
#     a = TLA.parse_logic_expression(inputs)
#     print(a)
