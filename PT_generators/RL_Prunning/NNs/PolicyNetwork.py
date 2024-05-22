import torch
from torch import nn

from PT_generators.RL_Prunning.Conifg import config

from PT_generators.RL_Prunning.NNs.Utility import getParFromModule


class PolicyNetwork(nn.Module):
    def __init__(self, ptg, get_program_feature):
        super().__init__()
        self.layer = nn.Linear(config.SIZE_EXP_NODE_FEATURE * 3, config.SIZE_EXP_NODE_FEATURE)
        self.ptg = ptg  # PT_generator 实例
        self.get_program_feature = get_program_feature # from symbol embedding

    # func 是这个
    # def GetProgramFearture(path2CFile, depth):
    #     problemID = path2CFile.split('/')[-1].split('.')[0]
    #     if 'NL' in problemID:
    #         problemStr = "Problem_NL" + problemID.split('NL')[-1]
    #     else:
    #         problemStr = "Problem_L" + problemID
    #     try:
    #         return SymbolEmbeddings[problemStr + "_" + str(depth)]
    #     except:
    #         return SymbolEmbeddings['?']
    def forward(self, state_vec, overall_feature):
        if torch.cuda.is_available():
            program_feature = self.get_program_feature(self.ptg.specname, self.ptg.depth).cuda()
        else:
            program_feature = self.get_program_feature(self.ptg.specname, self.ptg.depth)
        l1out = self.layer(torch.cat([state_vec, overall_feature, program_feature], 1))
        return l1out

    def get_parameters(self):
        res = {}
        prefix = "PolicyNetwork_P_"
        res.update(getParFromModule(self.layer, prefix=prefix + "layer"))
        return res

    def cudalize(self):
        self.layer = self.layer.cuda()
