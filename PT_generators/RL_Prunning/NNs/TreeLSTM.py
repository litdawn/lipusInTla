import logging

import torch
from torch import nn, tensor
from torch.nn import Parameter
from seedTemplate.tlaParser.tla import TLA
from seedTemplate.tlaParser.type import Type
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
    def __init__(self, seed_tmpl):  # vars 来自tla_ins.variables
        super().__init__()
        tla_ins = seed_tmpl.tla_ins
        self.rnns = {}
        self.tla_ins = tla_ins
        if config.use_self_generate:
            self.vars = tla_ins.variables
        else:
            self.vars = seed_tmpl.variables
        self.states = tla_ins.states
        # Lets give each sort of EXP an rnn
        a = '\\/,/\\,\\X'.split(',')
        b = '\\subseteq,\\in,[,('.split(',')
        c = '=,other'.split(',')
        self.keys = []
        self.keys.extend(a)
        self.keys.extend(b)
        self.keys.extend(c)
        for k in self.keys:
            self.rnns[k] = nn.LSTM(config.SIZE_EXP_NODE_FEATURE, config.SIZE_EXP_NODE_FEATURE, 2)

        self.attvec = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # Att1
        self.softmaxer = nn.Softmax(dim=1)

    # 整个表达式的特征
    '''
        state: 一个init/ind/next的名字
    '''

    def forward(self, s, state_or_seed):
        # if len(state.children()) > 0:

        if state_or_seed == "state":
            try:
                content = self.states[s].concrete_content
                if len(content) == 0:
                    state_ele_list = []
                else:
                    state_ele_list = TLA.parse_logic_expression(content)
            except:
                state_ele_list = []
        else:
            s = s.replace(",", " ").replace("<", "").replace(">", "").replace("(", "").replace(")", "")
            state_ele_list = s.split(" ")
        return self.find_and_concat(state_ele_list)

    def find_and_concat(self, state_ele_list):
        state_rnn = ""
        while len(state_ele_list) == 1:
            state_ele_list = state_ele_list[0]
        for k in self.keys:
            if k in set(state_ele_list):
                state_rnn = self.rnns[k]
                break
        if state_rnn == "":
            state_rnn = self.rnns["other"]
        # child_features = torch.ones((1, config.SIZE_EXP_NODE_FEATURE))  # 创建一个 初始化均为1 的张量

        child_feature_list = [torch.ones((1, config.SIZE_EXP_NODE_FEATURE))]
        # child_feature_list = []
        # for var in self.vars.keys():
        #     if var in state_ele_list:
        #         child_features = torch.cat((child_features, self.forward_var(var, state_ele_list)), 0).clone()

        for ele in state_ele_list:
            if not type(ele) == str:
                child_feature_list.append(self.find_and_concat(ele))
            elif ele in self.vars or ele in self.tla_ins.constants or ele in self.tla_ins.variables:
                child_feature_list.append(self.forward_var(ele))

        child_features = torch.cat(child_feature_list, dim=0)
        if torch.cuda.is_available():
            child_features = child_features.cuda()
        feature, _ = state_rnn(child_features.reshape([-1, 1, config.SIZE_EXP_NODE_FEATURE]))
        return feature[-1]

    def forward_var(self, var):
        if config.use_self_generate:
            if var.self_type == Type.BOOL:
                name = "bool"
            elif var.self_type == Type.ARRAY:
                name = "array"
            elif var.self_type == Type.SET:
                name = "set"
            elif var.self_type == Type.STRING:
                name = "str"
            else:
                name = "?"
        else:
            name = var
        origin = SymbolEmbeddings[name]
        return origin

    # init next ind exp的特征
    def forward_three(self, init_exp, next_exp, ind_exp):
        init_emb = self.forward(init_exp, "state")
        next_emb = self.forward(next_exp, "state")
        ind_emb = self.forward(ind_exp, "state")

        weis = torch.cat([torch.cosine_similarity(init_emb, self.attvec),
                          torch.cosine_similarity(next_emb, self.attvec),
                          torch.cosine_similarity(ind_emb, self.attvec)], 0).reshape([1, 3])
        swis = self.softmaxer(weis)
        # 计算余弦相似度，再转换为概率
        three_emb = torch.cat((init_emb, next_emb, ind_emb), 0).reshape([3, config.SIZE_EXP_NODE_FEATURE])
        smt_emb = torch.mm(swis, three_emb)  # 矩阵乘法
        return smt_emb

    def get_parameters(self):
        res = {}
        prefix = "Tree_LSTM_P_"
        res[prefix + "attvec"] = self.attvec
        for ky in self.rnns.keys():
            res.update(getParFromModule(self.rnns[ky], prefix=prefix + str(ky)))
        return res

    def cudalize(self):
        self.attvec = Parameter(self.attvec.cuda())
        for ky in self.rnns.keys():
            self.rnns[ky] = self.rnns[ky].cuda()

# # # littel Test
# if __name__ == "__main__":
