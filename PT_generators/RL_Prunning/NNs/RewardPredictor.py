import torch
from torch import nn, tensor

from PT_generators.RL_Prunning.Conifg import config
from PT_generators.RL_Prunning.NNs.Utility import getParFromModule


class RewardPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(config.SIZE_EXP_NODE_FEATURE * 2, config.SIZE_EXP_NODE_FEATURE)
        self.layer2 = nn.Linear(config.SIZE_EXP_NODE_FEATURE, config.SIZE_EXP_NODE_FEATURE // 2)
        self.layer3 = nn.Linear(config.SIZE_EXP_NODE_FEATURE // 2, 1)

    def forward(self, stateVec, overall_feature):
        tensorflow = tensor(torch.cat([stateVec, overall_feature], 1))
        if torch.cuda.is_available():
            tensorflow = tensorflow.cuda()
        l1out = self.layer1(tensorflow)
        m10 = tensor([[-10]])
        p10 = tensor([[10]])
        if torch.cuda.is_available():
            m10 = m10.cuda()
            p10 = p10.cuda()
        return torch.min(torch.cat([torch.max(torch.cat([self.layer3(self.layer2(l1out)), m10], 1)).reshape(1,1), p10], 1)).reshape(1,1)

    def get_parameters(self):
        res = {}
        prefix = "RewardPredictor_P_"
        res.update(getParFromModule(self.layer1, prefix=prefix + "layer1"))
        res.update(getParFromModule(self.layer2, prefix=prefix + "layer2"))
        res.update(getParFromModule(self.layer3, prefix=prefix + "layer3"))

        return res

    def cudalize(self):
        self.layer1 = self.layer1.cuda()
        self.layer2 = self.layer2.cuda()
        self.layer3 = self.layer3.cuda()


# little test

if __name__ == "__main__":
    stateVec = torch.randn([1, config.SIZE_EXP_NODE_FEATURE])
    overall_feature = torch.randn([1, config.SIZE_EXP_NODE_FEATURE])
    rp = RewardPredictor()
    print(rp(stateVec, overall_feature))


# 这是一个名为 RewardPredictor 的 PyTorch 类。这个类似乎是在某种上下文中预测奖励的模型，可能是一个强化学习场景。
#
# RewardPredictor 类有三层：
#
# layer1 是一个线性层，接收大小为 config.SIZE_EXP_NODE_FEATURE * 2 的输入，并输出大小为 config.SIZE_EXP_NODE_FEATURE 的输出。
# layer2 是一个线性层，接收大小为 config.SIZE_EXP_NODE_FEATURE 的输入，并输出大小为 config.SIZE_EXP_NODE_FEATURE // 2 的输出。
# layer3 是一个线性层，接收大小为 config.SIZE_EXP_NODE_FEATURE // 2 的输入，并输出大小为 1 的输出。
# 在 forward 方法中，首先将 stateVec 和 overall_feature 拼接在一起，然后通过 layer1、layer2 和 layer3 进行一系列的线性变换。
# 最后，输出的结果被限制在 -10 和 10 之间。
#
# GetParameters 方法返回模型的参数。
#
# cudalize 方法将模型的所有层都移动到 GPU 上。
#
# 在主函数中，创建了一个 RewardPredictor 对象，并对其进行了测试。测试中使用了随机生成的 stateVec 和 overall_feature。