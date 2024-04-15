import json

import torch
from torch import nn, tensor
from torch.nn import Parameter

# from PT_generators.RL_Prunning.Conifg import config
# from PT_generators.RL_Prunning.ExternalProcesses.CFG_parser import GetAllCGraphfilePath
# # from PT_generators.RL_Prunning.NNs.Utility import getParFromModule
# from code2inv.common.ssa_graph_builder import ProgramGraph
#
# from code2inv.graph_encoder.embedding import EmbedMeanField
# from code2inv.prog_generator.file_solver import GraphSample
#
#
# class CFG_Embedding(nn.Module):
#     # CFG_Embedding类的定义：
#     #
#     # 这是一个继承自torch.nn.Module的类，用于构建一个特定的神经网络模型。
#     # 在__init__构造函数中，首先通过一系列操作建立了一个节点类型字典（node_type_dict），
#     # 然后使用这个字典初始化了一些网络层和参数，
#     # 比如EmbedMeanField编码器、
#     # 一个可训练的向量attvec、
#     # 一个softmax层softmaxer，
#     # 以及一个GraphSample对象g_list
#     def __init__(self, cfg):
#         super().__init__()
#
#         # Need to prepare node type dict from the beginning.
#         node_type_dict = {}
#         allgpaths = GetAllCGraphfilePath()
#         for gpath in allgpaths:
#             graph_file = open(gpath, 'r')
#             graph = ProgramGraph(json.load(graph_file))
#             for node in graph.node_list:
#                 if not node.node_type in node_type_dict:
#                     v = len(node_type_dict)
#                     node_type_dict[node.node_type] = v
#
#         # 均值场
#         self.encoder = EmbedMeanField(config.SIZE_EXP_NODE_FEATURE, len(node_type_dict), max_lv=10)
#         # 待优化参数
#         self.attvec = Parameter(torch.randn((1,config.SIZE_EXP_NODE_FEATURE)), requires_grad=True) #Att3
#         # softmax
#         self.softmaxer = nn.Softmax(dim=1)
#         self.g_list = GraphSample(cfg, [], node_type_dict)
#
#     # 编码图形： self.cfg_emb = self.encoder(self.g_list)
#     # 这行代码使用前面定义的EmbedMeanField编码器对图形样本self.g_list进行编码。
#     # self.g_list包含了图形数据和节点类型信息。编码结果self.cfg_emb是图形中每个节点的嵌入表示。
#
#     # 计算加权特征： weighted1 = torch.mm(self.cfg_emb, stateVec.transpose(0, 1)).transpose(0, 1)
#     # 这里，torch.mm表示矩阵乘法。
#     # stateVec是传入的状态向量，可能代表了某种外部信息或条件。
#     # 首先，stateVec被转置以匹配self.cfg_emb的维度，
#     # 然后通过矩阵乘法与之相乘，得到加权后的特征。这一步的目的是根据状态向量调整图形嵌入的重要性。
#     #
#     # 合并和相似度计算： 接下来，使用torch.cosine_similarity计算cfg_emb、emb_smt和emb_CE与self.attvec（一个训练参数，代表注意力向量）之间的余弦相似度。
#     # emb_smt和emb_CE是外部传入的嵌入向量，分别代表了不同的信息维度。这一步骤生成了一个包含三个相似度值的向量。
#     #
#     # 应用Softmax： swis = self.softmaxer(weis)
#     # 使用Softmax层对上一步得到的相似度向量进行归一化，得到每个维度的权重。这一步骤确保了权重的总和为1，使得模型可以根据相似度自适应地调整每个特征的影响力。
#     #
#     # 特征融合： three_emb = torch.cat((cfg_emb, emb_smt, emb_CE), 0).reshape([3, config.SIZE_EXP_NODE_FEATURE])
#     # 将三种不同的嵌入（图形嵌入、emb_smt和emb_CE）合并为一个矩阵。
#     # 然后，overall_feature = torch.mm(swis, three_emb)通过矩阵乘法将Softmax权重swis应用到合并后的嵌入上，实现了特征的加权融合。这一步的结果
#     # overall_feature是一个综合考虑了三种不同信息源的特征表示。
#     #
#     # 输出： 最后，forward
#     # 方法返回
#     # overall_feature，这是经过编码、加权、相似度计算和融合的综合特征向量，可以用于后续的处理或作为模型的最终输出。
#
#     def forward(self, emb_smt, emb_CE, stateVec):
#         self.cfg_emb = self.encoder(self.g_list)
#         weighted1 = torch.mm(self.cfg_emb, stateVec.transpose(0,1)).transpose(0,1)
#         cfg_emb = torch.mm(weighted1, self.cfg_emb)
#         weis = torch.cat([torch.cosine_similarity(cfg_emb, self.attvec),
#                        torch.cosine_similarity(emb_smt, self.attvec),
#                        torch.cosine_similarity(emb_CE, self.attvec)],0).reshape([1,3])
#         swis = self.softmaxer(weis)
#         three_emb = torch.cat((cfg_emb, emb_smt, emb_CE), 0).reshape([3, config.SIZE_EXP_NODE_FEATURE])
#         overall_feature = torch.mm(swis, three_emb)
#         return overall_feature
#
#     def GetParameters(self):
#         res = {}
#         PreFix = "CFG_Embedding_P_"
#         res[PreFix + "attvec"] = self.attvec
#         res.update(getParFromModule(self.encoder, prefix=PreFix + "encoder"))
#         return res
#
#
#     def cudalize(self):
#         self.attvec = Parameter(self.attvec.cuda())
#         self.encoder = self.encoder.cuda()