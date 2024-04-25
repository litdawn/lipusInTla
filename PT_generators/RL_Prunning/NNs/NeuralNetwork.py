import pickle

import torch
from torch import tensor
from torch.nn import Parameter

from PT_generators.RL_Prunning.Conifg import config
from PT_generators.RL_Prunning.NNs.CounterExampleEmbedding import CEEmbedding
from PT_generators.RL_Prunning.NNs.OverallEmbedding import OverallEmbedding
from PT_generators.RL_Prunning.NNs.DistributionLize import DistributionLize
from PT_generators.RL_Prunning.NNs.IntLize import IntLize
from PT_generators.RL_Prunning.NNs.PolicyNetwork import PolicyNetwork
from PT_generators.RL_Prunning.NNs.RewardPredictor import RewardPredictor
from PT_generators.RL_Prunning.NNs.SymbolEmbeddings import SymbolEmbeddings
from PT_generators.RL_Prunning.NNs.TreeLSTM import TreeLSTM
from PT_generators.RL_Prunning.Template.Seed2Lemma import RULE


def constructT(vars):
    treeLSTM = TreeLSTM(vars)
    return treeLSTM


def constructG():
    return OverallEmbedding()


def constructE(vars):
    return CEEmbedding(vars)


def constructP():
    return RewardPredictor()


def constructpi(ptg):
    return PolicyNetwork(ptg, GetProgramFearture)


def construct_distributionlize():
    return DistributionLize()


def construct_intValuelzie():
    return IntLize()


def init_symbolEmbeddings():
    Rule_keys = RULE.keys()
    for non_terminal in Rule_keys:
        SymbolEmbeddings[non_terminal] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
        actions = RULE[non_terminal]
        for act in actions:
            SymbolEmbeddings[str(act)] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)


def GetProgramFearture(path2tla, depth):
    problem_name = path2tla.split('/')[-1].split('.')[0]
    try:
        return SymbolEmbeddings[problem_name]
    except:
        SymbolEmbeddings[problem_name] = Parameter(torch.randn((1, config.SIZE_EXP_NODE_FEATURE)), requires_grad=True)
        return SymbolEmbeddings[problem_name]


def GPUlizeSymbols():
    for keyname in SymbolEmbeddings.keys():
        SymbolEmbeddings[keyname] = Parameter(SymbolEmbeddings[keyname].cuda())


# def initialize_paramethers(path):
#     if "NL" in path:
#         ppPath = r"code2inv/templeter/NL_initial.psdlf"
#     else:
#         ppPath = r"code2inv/templeter/L_initial.psdlf"
#     with open(ppPath, 'rb') as f:
#         dict = pickle.load(f)
#         return dict


def GetActionIndex(last_left_handle, last_action):
    for i, action in enumerate(RULE[str(last_left_handle)]):
        if str(action) == str(last_action):
            if torch.cuda.is_available():
                return tensor([i]).cuda()
            else:
                return tensor([i])

    assert False  # should not be here

# constructT()：这个函数用于构建一个树状长短期记忆网络（TreeLSTM）。TreeLSTM是一种特殊的递归神经网络，它能够在树形结构数据上进行高效的特征学习。
# constructG(cfg)：这个函数接收一个控制流图（Control Flow Graph，CFG）作为输入，返回一个CFG的嵌入表示。这个嵌入表示可以捕获CFG的结构和属性，用于后续的学习任务。
# constructE(vars)：这个函数接收一组变量作为输入，返回一个反例（Counter Example）的嵌入表示。这个嵌入表示可以捕获反例的特性，用于后续的学习任务。
# constructP()：这个函数用于构建一个奖励预测器（Reward Predictor）。奖励预测器是强化学习中的一个重要组件，它用于预测每个动作的奖励值，以指导智能体的行为。
# constructpi(ptg)：这个函数接收一个程序模板生成器（Program Template Generator，PTG）作为输入，返回一个策略网络（Policy Network）。策略网络是强化学习中的一个重要组件，它用于根据当前状态选择动作。
# construct_distributionlize()：这个函数用于构建一个分布化函数。这个函数可以将一组数值转化为一个概率分布，用于后续的学习任务。
# construct_intValuelzie()：这个函数用于构建一个整数化函数。这个函数可以将一组数值转化为整数，用于后续的学习任务。
# init_symbolEmbeddings()：这个函数用于初始化符号嵌入。符号嵌入是一种将符号（如单词、字符等）转化为稠密向量的技术，它可以捕获符号的语义和语法信息。
# GetProgramFearture(path2CFile, depth)：这个函数接收一个文件路径和一个深度值作为输入，返回该程序的特征表示。
# GPUlizeSymbols()：这个函数用于将符号嵌入转移到GPU。这通常用于加速计算。
# initialize_paramethers(path)：这个函数接收一个路径作为输入，返回该路径下的初始化参数。
# GetActionIndex(last_left_handle,last_action)：这个函数接收一个句柄和一个动作作为输入，返回该动作在规则集中的索引。
