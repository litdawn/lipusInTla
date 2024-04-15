import torch
from torch import nn, tensor
from torch.nn import Parameter

from PT_generators.RL_Prunning.Conifg import *

from PT_generators.RL_Prunning.NNs.SymbolEmbeddings import SymbolEmbeddings
from PT_generators.RL_Prunning.NNs.Utility import getParFromModule

# 原forward,做参考
# def forward(self, z3_exp):
#     if len(z3_exp.children()) > 0:
#         k = str(z3_exp.decl())
#         Rnn = self.RNNS[k]
#         child_feartures = torch.ones((1, config.SIZE_EXP_NODE_FEATURE))  # 创建一个 初始化均为1 的张量
#         if torch.cuda.is_available():
#             child_feartures = child_feartures.cuda()
#         for chi in z3_exp.children():
#             child_feartures = torch.cat((child_feartures, self.forward(chi)), 0)
#         feature, _ = Rnn(child_feartures.reshape([-1, 1, config.SIZE_EXP_NODE_FEATURE]))
#         return feature[-1]
#     else:
#         if str(z3_exp.decl()) in SymbolEmbeddings:
#             return SymbolEmbeddings[str(z3_exp.decl())]  # 为每个children创建一个可梯度的
#         else:
#             return SymbolEmbeddings['?']

class TreeLSTM(nn.Module):
    def __init__(self, tla_ins):  # vars 来自tla_ins.variables
        super().__init__()
        self.RNNS = {}
        self.tla_ins = tla_ins
        self.vars = tla_ins.vars
        self.states = tla_ins.states
        # Lets give each sort of Z3 EXP an rnn
        a = '\\cup,\\cap,\\X'.split(',')
        b = '\\subseteq,\\in,[,('.split(',')
        c = '=,other'
        keys = []
        keys.extend(a)
        keys.extend(b)
        keys.extend(c)
        for k in keys:
            self.RNNS[k] = nn.LSTM(config.SIZE_EXP_NODE_FEATURE, config.SIZE_EXP_NODE_FEATURE, 2)

        self.attvec = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # Att1
        self.softmaxer = nn.Softmax(dim=1)

    # 整个表达式的特征
    '''
        state: 一个init/ind/next的名字
    '''

    def forward_state(self, state):
        # if len(state.children()) > 0:
        state_rnn = ""
        # self.tla_ins.
        for k in self.keys:
            if k in state:
                state_rnn = self.RNNS[k]
        if state_rnn == "":
            state_rnn = self.RNNS["other"]
        child_feartures = torch.ones((1, config.SIZE_EXP_NODE_FEATURE))  # 创建一个 初始化均为1 的张量
        if torch.cuda.is_available():
            child_feartures = child_feartures.cuda()
        for var in self.vars.keys():
            if var in self.states[state]:
                child_feartures = torch.cat((child_feartures, self.forward_var(var)), 0)
        feature, _ = state_rnn(child_feartures.reshape([-1, 1, config.SIZE_EXP_NODE_FEATURE]))
        return feature[-1]
    # todo symbolembedding相关 nn的初始化 tc的选择策略更新 var
    def forward_var(self, var):
        return SymbolEmbeddings['?']

    # init next ind exp的特征
    def forward_three(self, init_exp, next_exp, ind_exp):
        init_emb = self.forward_state(init_exp)
        next_emb = self.forward_state(next_exp)
        ind_emb = self.forward_state(ind_exp)

        weis = torch.cat([torch.cosine_similarity(init_emb, self.attvec),
                          torch.cosine_similarity(next_emb, self.attvec),
                          torch.cosine_similarity(ind_emb, self.attvec)], 0).reshape([1, 3])
        swis = self.softmaxer(weis)
        # 计算余弦相似度，再转换为概率
        three_emb = torch.cat((init_emb, next_emb, ind_emb), 0).reshape([3, config.SIZE_EXP_NODE_FEATURE])
        smt_emb = torch.mm(swis, three_emb)  # 矩阵乘法
        return smt_emb

    def GetParameters(self):
        res = {}
        PreFix = "Tree_LSTM_P_"
        res[PreFix + "attvec"] = self.attvec
        for ky in self.RNNS.keys():
            res.update(getParFromModule(self.RNNS[ky], prefix=PreFix + str(ky)))
        return res

    def cudalize(self):
        self.attvec = Parameter(self.attvec.cuda())
        for ky in self.RNNS.keys():
            self.RNNS[ky] = self.RNNS[ky].cuda()


# # littel Test
# if __name__ == "__main__":
#     from pyparsing import Forward, Combine, infixNotation, opAssoc, Keyword, Word, alphanums, Suppress, Optional,ZeroOrMore
#
#     # 定义操作数、函数名和符号
#     identifier = Word(alphanums + "_")
#     function_name = Word(alphanums + "_")
#     keyword_not = Keyword("~")
#     keyword_subset = Keyword("\\subseteq")
#     keyword_belongs_to = Keyword("\\in")
#     keyword_and = Keyword("/\\")
#     keyword_or = Keyword("\\/")
#
#     # 定义括号
#     LPAREN = Suppress("(")
#     RPAREN = Suppress(")")
#
#     # 定义操作符优先级
#     precedence = [
#         (keyword_subset, 2, opAssoc.LEFT),
#         (keyword_belongs_to, 2, opAssoc.LEFT),
#         (keyword_and, 2, opAssoc.LEFT),
#         (keyword_or, 2, opAssoc.LEFT),
#
#     ]
#
#     # 定义逻辑表达式
#     expr = Forward()
#     atom = Forward()
#     identifiers = Combine(
#         "<<" + Optional(keyword_not) + identifier + ZeroOrMore("," + Optional(keyword_not) + identifier) + ">>")
#
#     # 定义函数参数列表
#     arg = (identifier | identifiers) | function_name | "~" + identifier
#     args_list = arg + ZeroOrMore("," + arg)
#
#     # 定义函数调用表达式
#     function_call = Combine(function_name + "(" + Optional(args_list) + ")")
#     atom <<= "~" + LPAREN + expr + RPAREN | LPAREN + expr + RPAREN | function_call | identifiers | identifier
#     expr <<= infixNotation(atom, precedence)
#
#
#     # 解析逻辑表达式
#     def parse_logic_expression(expression):
#         return expr.parseString(expression)
#
#
#     logic_expression = "(ResponseMatched(VARI,VARP)) \\/ (~(<<VARI,VARP>> \\in response_sent))"
#     parsed_logic = parse_logic_expression(logic_expression)
#     print(parsed_logic)