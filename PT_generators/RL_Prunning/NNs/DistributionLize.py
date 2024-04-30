import torch
from torch import nn

from PT_generators.RL_Prunning.NNs.SymbolEmbeddings import SymbolEmbeddings


class DistributionLize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, action_vector, seeds_vec):
        # construt the available action vectors
        rawness = torch.cat(
            [torch.mm(x, action_vector.transpose(0, 1)) for x in seeds_vec],
            1)
        likenesses = torch.softmax(rawness, 1)
        return likenesses, rawness

    def get_parameters(self):
        res = {}

        return res
