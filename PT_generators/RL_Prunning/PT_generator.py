from torch.optim import Adam

from PT_generators.RL_Prunning.ExternalProcesses.CFG_parser import parseCFG
from PT_generators.RL_Prunning.ExternalProcesses.SMT_parser import parseSMT
from PT_generators.RL_Prunning.ExternalProcesses.Sampling import sampling
from PT_generators.RL_Prunning.NNs.NeuralNetwork import *
from PT_generators.RL_Prunning.TemplateCenter.TemplateCenter import InitPT, getLeftHandle, init_varSelection, \
    AvailableActionSelection, update_PT_rule_selction, ShouldStrict, StrictnessDirtribution, simplestAction, \
    init_constSelection, LossnessDirtribution
from seedTemplate.tlaParser.tlaparser import get_varnames_from_source_code, get_consts_from_source_code
import torch.nn.functional as F


class PT_generator:

    #
    # 解析输入：使用parseCFG和parseSMT函数解析输入的CFG和SMT文件，并将结果分别存储在self.cfg和self.smt中。同时，从源代码中获取变量名和常量，并分别存储在self.vars和self.consts中。
    # 初始化选择：使用init_varSelection，init_constSelection和init_symbolEmbeddings函数初始化变量选择，常量选择和符号嵌入。
    # 构造神经网络并加载参数：使用constructT，constructG，constructE，constructP，constructpi，construct_distributionlize函数构造各个神经网络并加载参数。如果可以使用GPU，则调用self.gpulize()
    # 将神经网络转移到GPU上。
    # 初始化学习器和参数：调用self.init_learner_par()
    # 和self.init_parameters()
    # 初始化学习器和参数。
    def __init__(self):
        self.LR = config.LearningRate
        # Step1. Parse the inputs
        # self.cfg = parseCFG(path2CFG)
        # self.smt = parseSMT(path2SMT)
        # self.path2CFile = path2CFile
        # self.vars = get_varnames_from_source_code(self.path2CFile)
        # self.consts = get_consts_from_source_code(self.path2CFile)

        init_varSelection(self.vars)

        init_constSelection(self.consts)

        init_symbolEmbeddings()

        # Step2. Construct the NNs and Load the parameters
        # 在 NNs.NeuralNetwork 里
        self.T = constructT()  # treeLSTM
        # self.G = constructG(self.cfg)  # CFG_Embedding
        self.E = constructE(self.vars)  # CounterExample_Embedding
        self.P = constructP()  # reward predictor
        self.pi = constructpi(self)  # policyNetwork
        self.distributionlize = construct_distributionlize()  # DistributionLize()
        # self.intValuelzie = construct_intValuelzie()

        # if we can use gpu
        if torch.cuda.is_available():
            self.gpulize()

        # Step3. Init the learner and parameters
        self.init_learner_par()

        # if config.CONTINUE_TRAINING:
        #     self.load_parameters(config.ppath)
        self.init_parameters()

    # 它的主要功能是生成下一个程序树（Program Tree，PT）。以下是每个步骤的解释：
    #
    # 初始化：设置深度为0，初始化程序树PT，计算状态向量self.stateVec，并初始化一些将在后续使用的列表。
    # 嵌入代码元素：使用self.E(CE)
    # 和self.T.forward_three(self.smt)
    # 将代码元素和SMT公式嵌入到向量空间，并分别存储在emb_CE和self.emb_smt中。
    # 处理程序树：在程序树PT的左句柄不为空的情况下，进行以下操作：
    # 获取可用的动作和动作或值的类型。
    # 计算整体特征和预测的奖励。
    # 如果需要选择一个动作，那么就从可用的动作中采样一个动作，并更新程序树。如果深度超过了最大深度，那么就选择最简单的动作。
    # 如果需要选择一个值，那么就从值的分布中采样一个值，并更新程序树。但是请注意，这部分代码目前被注释掉了，所以实际上并不会执行。
    # 更新状态向量和深度：使用新的程序树更新状态向量，并将深度加1。
    # 保存结果：将预测的奖励列表、选择的动作列表、输出的列表、动作或值的类型列表和左句柄列表保存到类的属性中。
    # 返回结果：返回更新后的程序树PT。
    def generate_next(self, CE):
        self.depth = 0
        PT = InitPT()  # Bool('non_nc')
        # 1
        self.stateVec = self.T(PT)
        # the lists will be used when punish or prised.
        predicted_reward_list = []
        action_selected_list = []
        outputed_list = []
        action_or_value = []
        left_handles = []
        # CE = {'p': [],'n': [],'i': []}
        # 2
        emb_CE = self.E(CE)
        # pre_exp, trans_exp, post_exp这三个是self.smt
        # 3
        self.emb_smt = self.T.forward_three(self.smt)
        # 函数返回的是输入表达式 PT 中最左侧的句柄, non代表非终结符
        left_handle = getLeftHandle(PT)
        while left_handle is not None:
            left_handles.append(left_handle)
            # 在templatecenter的RULE里面随机选规则
            act_or_val, available_acts = AvailableActionSelection(left_handle)
            # 总体特征
            overall_feature = self.G(self.emb_smt, emb_CE, self.stateVec)
            predicted_reward = self.P(self.stateVec, overall_feature)
            predicted_reward_list.append(predicted_reward)
            action_vector = self.pi(self.stateVec, overall_feature)
            if act_or_val == config.SELECT_AN_ACTION:
                action_dirtibution, action_raw = self.distributionlize(action_vector, available_acts)
                action_selected = sampling(action_dirtibution, available_acts)

                if self.depth >= config.MAX_DEPTH:
                    action_selected = simplestAction(left_handle)
                action_selected_list.append(action_selected)
                outputed_list.append(action_raw)

                PT = update_PT_rule_selction(PT, left_handle, action_selected)

            else:
                assert False  # should not be here now
                # value = self.intValuelzie(action_vector, left_handle)
                # value_of_int = int(value)
                # action_selected_list.append(value_of_int)
                # outputed_list.append(value)
                #
                # PT = update_PT_value(PT, left_handle, value_of_int)

            action_or_value.append(act_or_val)
            left_handle = getLeftHandle(PT)
            self.stateVec = self.T(PT)
            self.depth += 1

        self.last_predicted_reward_list = predicted_reward_list
        self.last_action_selected_list = action_selected_list
        self.last_outputed_list = outputed_list
        self.last_action_or_value = action_or_value
        self.last_left_handles = left_handles
        return PT

    # 设置奖励和伽马值：根据Deg的值设置奖励和伽马值。
    # 如果Deg是"VERY"，那么奖励是 - 10，伽马值是0.1。
    # 如果Deg是 "MEDIUM"，那么奖励是 - 5，伽马值是0.05。
    # 如果Deg是 "LITTLE"，那么奖励是 - 1，伽马值是0.01。
    # 初始化严格损失：初始化严格损失为0。如果可以使用GPU，那么将严格损失转移到GPU上。
    # 计算严格损失：对于self.last_action_or_value中的每一个元素，如果它应该被严格处理（由ShouldStrict函数判断），那么就计算严格损失。
    #       如果这个动作是选择一个动作，那么就计算严格性分布或松散性分布，并计算严格性损失。
    #       如果这个动作是选择一个值，那么就计算均方误差损失（但是请注意，这部分代码目前被注释掉了，所以实际上并不会执行）。然后将严格损失累加到总的严格损失上，并将计数器加1。
    # 计算平均严格损失：如果计数器不为0，那么就将总的严格损失除以计数器，得到平均严格损失。
    # 计算动作损失并进行学习步骤：调用self.ALoss(reward)
    # 计算动作损失，然后调用self.LearnStep((a_loss + strict_loss))
    # 进行学习步骤。
    def punish(self, SorL, Deg, Whom):
        gama = 0
        reward = 0
        if Deg == "VERY":
            reward = -10
            gama = 0.1
        elif Deg == "MEDIUM":
            reward = -5
            gama = 0.05
        elif Deg == "LITTLE":
            reward = -1
            gama = 0.01
        assert gama != 0
        assert reward != 0

        strict_loss = tensor([[0]], dtype=torch.float32)
        if torch.cuda.is_available():
            strict_loss = strict_loss.cuda()
        counter = 0
        for i in range(len(self.last_action_or_value)):
            if ShouldStrict(self.last_left_handles[i], Whom):
                if self.last_action_or_value[i] == config.SELECT_AN_ACTION:
                    if SorL == 'STRICT':
                        SD = StrictnessDirtribution(self.last_left_handles[i], Whom)
                    else:
                        assert SorL == 'LOOSE'
                        SD = LossnessDirtribution(self.last_left_handles[i], Whom)
                    Loss_strictness = -torch.mm(SD, torch.log_softmax(self.last_outputed_list[i].reshape(1, -1),
                                                                      1).transpose(0,
                                                                                   1)) * gama
                else:
                    assert False  # should not be here
                    # Loss_strictness = F.mse_loss(self.last_outputed_list[i],
                    #                              torch.tensor([1], dtype=torch.float32)) * gama / 4

                strict_loss += Loss_strictness.reshape([1, 1])
                counter += 1
        if counter != 0:
            strict_loss /= counter
        a_loss = self.ALoss(reward)
        self.LearnStep((a_loss + strict_loss))

    # 计算奖励列表：根据final_reward和discounter计算每一步的奖励，并将结果存储在reward_list中。
    # 计算预测损失：对于self.last_predicted_reward_list中的每一个元素，计算预测的奖励和上一步的预测奖励，然后根据动作的类型计算损失。如果这个动作是选择一个动作，那么就计算交叉熵损失。如果这个动作是选择一个值，那么就计算均方误差损失（但是请注意，这部分代码目前被注释掉了，所以实际上并不会执行）。然后将损失乘以奖励和上一步的预测奖励的差值，并累加到总的预测损失上。
    # 计算平均预测损失：将总的预测损失除以奖励列表的长度，得到平均预测损失。
    # 计算均方误差损失：使用F.mse_loss函数计算奖励列表和预测奖励列表之间的均方误差损失。
    # 返回总损失：将预测损失和均方误差损失相加，得到总损失，并返回。

    def ALoss(self, final_reward):
        discounter = 0.95
        reward_list = []
        for i in range(len(self.last_predicted_reward_list)):
            reward_list.append(final_reward * discounter ** i)
        reward_list = reward_list[::-1]
        p_loss = 0
        for i in range(len(self.last_predicted_reward_list)):
            r_i = reward_list[i]
            if i == 0:
                pr_i_1 = tensor([[0]], dtype=torch.float32)
            else:
                # pr_i_1 = self.last_predicted_reward_list[i - 1]
                pr_i_1 = tensor([reward_list[i - 1]], dtype=torch.float32)
            if self.last_action_or_value[i] == config.SELECT_AN_ACTION:
                losser = F.cross_entropy(self.last_outputed_list[i].reshape(1, -1),
                                         GetActionIndex(self.last_left_handles[i], self.last_action_selected_list[i]))
            else:
                assert False
            if torch.cuda.is_available():
                p_loss += (tensor(r_i, dtype=torch.float32) - pr_i_1).cuda() * losser.reshape([1, 1])
            else:
                p_loss += (tensor(r_i, dtype=torch.float32) - pr_i_1) * losser.reshape([1, 1])
        p_loss = p_loss / len(reward_list)
        if torch.cuda.is_available():
            mse_loss = F.mse_loss(tensor([reward_list], dtype=torch.float32).cuda(),
                                  torch.cat(self.last_predicted_reward_list, 1)).reshape([1, 1])
        else:
            mse_loss = F.mse_loss(tensor([reward_list], dtype=torch.float32),
                                  torch.cat(self.last_predicted_reward_list, 1)).reshape([1, 1])
        # print("mse_loss", mse_loss)
        # print("p_loss", p_loss)
        return (p_loss + mse_loss)

    def prise(self, Deg):
        if Deg == "VERY":
            reward = 10
        elif Deg == "LITTLE":
            reward = 1
        else:
            reward = 0
        a_loss = self.ALoss(reward)
        self.LearnStep(a_loss)

    def LearnStep(self, loss):
        # if torch.cuda.is_available():
        #     loss = loss.cuda()
        self.adam.zero_grad()
        # print(loss)
        loss.backward()
        # if torch.cuda.is_available():
        #     loss = loss.cpu()
        self.adam.step()

    def init_learner_par(self):
        paras = {}
        paras.update(SymbolEmbeddings)
        paras.update(self.T.GetParameters())
        paras.update(self.G.GetParameters())
        paras.update(self.E.GetParameters())
        paras.update(self.P.GetParameters())
        paras.update(self.pi.GetParameters())
        # paras.update(self.distributionlize.GetParameters())
        # paras.update(self.intValuelzie.GetParameters())
        for parname in paras:
            paras[parname].requires_grad = True

        self.adam = Adam(paras.values(), lr=self.LR)
        self.paras = paras

    def init_parameters(self):
        paradict = initialize_paramethers(self.path2CFile)
        for parname in self.paras:
            if parname in paradict:  # initialize
                assert self.paras[parname].shape == paradict[parname].shape
                self.paras[parname].data = paradict[parname].data

    def gpulize(self):
        self.T.cudalize()
        self.G.cudalize()
        self.E.cudalize()
        self.P.cudalize()
        self.pi.cudalize()
        GPUlizeSymbols()
