import string

import torch
from torch import nn
from torch.nn import Parameter

from PT_generators.RL_Prunning.Conifg import *

SymbolEmbeddings = {}
# for word in string.ascii_lowercase:  # variables
#     SymbolEmbeddings[word] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)

# SymbolEmbeddings['Int'] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # const
# 需要梯度计算的张量。该张量是模型参数，需要根据反向传播算法进行更新。
SymbolEmbeddings['?'] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # default
SymbolEmbeddings['str'] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
SymbolEmbeddings['bool'] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
SymbolEmbeddings['set'] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
SymbolEmbeddings['array'] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)

SymbolEmbeddings['Rule_and'] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
SymbolEmbeddings['Rule_or'] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
SymbolEmbeddings['Rule_neg'] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)

# SymbolEmbeddings['in'] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
# SymbolEmbeddings['[]'] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
# SymbolEmbeddings['subseteq'] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
# SymbolEmbeddings['()'] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
# SymbolEmbeddings['='] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)


# SymbolEmbeddings['non_v'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # non-determinals
# SymbolEmbeddings['non_s'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # non-determinals
# SymbolEmbeddings['non_p'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # non-determinals
# SymbolEmbeddings['non_op1'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # non-determinals
# SymbolEmbeddings['non_op2'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # non-determinals
# SymbolEmbeddings['non_t'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # non-determinals
# SymbolEmbeddings['non_nd'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)),
#                                             requires_grad=True)  # non-determinals
# SymbolEmbeddings['non_nc'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)),
#                                       requires_grad=True)  # non-determinals
#
#
#
# SymbolEmbeddings['Rule_p_<'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # rule
# SymbolEmbeddings['Rule_p_<='] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # rule
# SymbolEmbeddings['Rule_p_>'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # rule
# SymbolEmbeddings['Rule_p_>='] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # rule
# SymbolEmbeddings['Rule_p_=='] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # rule
#
# SymbolEmbeddings['Rule_op2_+'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # rule
# SymbolEmbeddings['Rule_op2_-'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # rule
# SymbolEmbeddings['Rule_op2_*'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # rule
# SymbolEmbeddings['Rule_op2_/'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # rule
# SymbolEmbeddings['Rule_op2_%'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # rule
#
# SymbolEmbeddings['Rule_op1_-'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # rule
# SymbolEmbeddings['Rule_op1_abs'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # rule
#
# SymbolEmbeddings['Rule_t_v'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # rule
# SymbolEmbeddings['Rule_t_s'] = Variable(torch.zeros((1, SIZE_EXP_NODE_FEATURE)), requires_grad=True)  # rule
# SymbolEmbeddings['Rule_t_op2'] = SymbolEmbeddings['non_op2']  # rule
# SymbolEmbeddings['Rule_t_op1'] = SymbolEmbeddings['non_op1']  # rule

