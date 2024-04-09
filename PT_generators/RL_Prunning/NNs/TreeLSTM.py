import torch
from torch import nn, tensor
from torch.nn import Parameter

from PT_generators.RL_Prunning.Conifg import *

from PT_generators.RL_Prunning.NNs.SymbolEmbeddings import SymbolEmbeddings
from PT_generators.RL_Prunning.NNs.Utility import getParFromModule


class TreeLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.RNNS = {}
        # Lets give each sort of Z3 EXP an rnn
        ops = '+,-,*,/,%,If'.split(',')
        cps = '<,>,<=,>=,=='.split(',')
        lcs = 'Not,And,Or,Implies'.split(',')
        keys = []
        keys.extend(ops)
        keys.extend(cps)
        keys.extend(lcs)
        for k in keys:
            self.RNNS[k] = nn.LSTM(config.SIZE_EXP_NODE_FEATURE, config.SIZE_EXP_NODE_FEATURE, 2)

        self.attvec = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # Att1
        self.softmaxer = nn.Softmax(dim=1)

    # 整个表达式的特征
    def forward(self, z3_exp):
        if len(z3_exp.children()) > 0:
            k = str(z3_exp.decl())
            Rnn = self.RNNS[k]
            child_feartures = torch.ones((1, config.SIZE_EXP_NODE_FEATURE))  # 创建一个 初始化均为1 的张量
            if torch.cuda.is_available():
                child_feartures = child_feartures.cuda()
            for chi in z3_exp.children():
                child_feartures = torch.cat((child_feartures, self.forward(chi)), 0)
            feature, _ = Rnn(child_feartures.reshape([-1, 1, config.SIZE_EXP_NODE_FEATURE]))
            return feature[-1]
        else:
            if str(z3_exp.decl()) in SymbolEmbeddings:
                return SymbolEmbeddings[str(z3_exp.decl())] # 为每个children创建一个可梯度的
            else:
                return SymbolEmbeddings['?']

    # pre/trans/post exp的特征
    def forward_three(self, args):
        pre_exp, trans_exp, post_exp = args
        pre_emb = self.forward(pre_exp)
        trans_emb = self.forward(trans_exp)
        post_emb = self.forward(post_exp)

        weis = torch.cat([torch.cosine_similarity(pre_emb, self.attvec),
                       torch.cosine_similarity(trans_emb, self.attvec),
                       torch.cosine_similarity(post_emb, self.attvec)], 0).reshape([1, 3])
        swis = self.softmaxer(weis)
        # 计算余弦相似度，再转换为概率
        three_emb = torch.cat((pre_emb, trans_emb, post_emb), 0).reshape([3, config.SIZE_EXP_NODE_FEATURE])
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


# littel Test
if __name__ == "__main__":
    TR = TreeLSTM()
    from z3 import *

    x = Int('x')
    y = Int('y')

    ee1 = And(x + y < 3, x * (x % y + 2) >= Z3_abs(x * y))
    ee2 = Or(x + y < 3, x * (x % y + 2) >= Z3_abs(x * y))
    ee3 = x * (x % y + 2) >= Z3_abs(x * y)
    pp = TR.forward_three((ee1, ee2, ee3))
    print(pp)

# 在 __init__ 方法中，它初始化了一些 LSTM 网络，每种 Z3 表达式都有一个 LSTM 网络。它还初始化了一个参数 self.attvec，这个参数是用于计算余弦相似度的。
#
# 在 forward 方法中，如果 z3_exp 有子节点，那么它会对每个子节点调用 forward 方法，并将结果连接起来。然后，它会将这个结果通过 LSTM 网络进行处理，并返回最后一个特征。如果 z3_exp 没有子节点，那么它会返回对应的符号嵌入。
#
# forward_three 方法接受三个参数，分别是 pre_exp、trans_exp 和 post_exp。它会对这三个参数分别调用 forward 方法，并将结果连接起来。然后，它会计算这个结果与 self.attvec 的余弦相似度，并通过 softmax 函数进行归一化。最后，它会返回这个结果与三个嵌入的矩阵乘积。
#
# GetParameters 方法会返回一个字典，这个字典包含了 self.attvec 和所有 LSTM 网络的参数。
#
# cudalize 方法会将 self.attvec 和所有 LSTM 网络移动到 GPU 上。