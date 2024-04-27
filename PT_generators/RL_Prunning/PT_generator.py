from torch.optim import Adam
from PT_generators.RL_Prunning.NNs.NeuralNetwork import *
from PT_generators.RL_Prunning.Template.Seed2Lemma import *
import torch.nn.functional as F
import PT_generators.RL_Prunning.Conifg

class PT_generator:

    #
    # 解析输入：使用parseCFG和parseSMT函数解析输入的CFG和SMT文件，并将结果分别存储在self.cfg和self.smt中。同时，从源代码中获取变量名和常量，并分别存储在self.vars和self.consts中。
    # 初始化选择：使用init_varSelection，init_constSelection和init_symbolEmbeddings函数初始化变量选择，常量选择和符号嵌入。
    # 构造神经网络并加载参数：使用constructT，constructG，constructE，constructP，constructpi，construct_distributionlize函数构造各个神经网络并加载参数。如果可以使用GPU，则调用self.gpulize()
    # 将神经网络转移到GPU上。
    # 初始化学习器和参数：调用self.init_learner_par()
    # 和self.init_parameters()
    # 初始化学习器和参数。
    def __init__(self, seed_tmpl):
        self.emb_tla = None
        self.adam = None
        self.paras = None
        self.stateVec = dict()  # key: lemma_name val:lemma_tensor

        self.depth = 0
        self.LR = config.LearningRate
        # Step1. Parse the inputs
        self.seed_tmpl = seed_tmpl

        init_symbolEmbeddings()

        # Step2. Construct the NNs and Load the parameters
        # 在 NNs.NeuralNetwork 里
        self.T = constructT(seed_tmpl.tla_ins)  # treeLSTM
        self.G = constructG()  # CFG_Embedding
        self.E = constructE(seed_tmpl.seeds)  # CounterExample_Embedding
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
        # self.init_parameters()

        self.candidate = dict()
        self.lemma_pointer = 0
        self.init_candidate()

    def init_candidate(self):
        self.candidate.clear()
        self.candidate.update({"Safety": self.seed_tmpl.tla_ins.inv})
        self.candidate.update({"Typeok": self.seed_tmpl.tla_ins.type_ok})

    def generate_next(self, CE):
        self.depth = 0

        # 1 初始化
        candidate = []
        # the lists will be used when punish or prised.
        predicted_reward_list = dict()
        action_selected_list = []
        outputed_list = []
        action_or_value = []

        # 2 embedding counter example, state
        emb_ce = self.E(CE)
        tla_ins = self.seed_tmpl.tla_ins
        self.emb_tla = self.T.forward_three(tla_ins.init, tla_ins.next, tla_ins.inv)

        # 3. 嵌入seed
        for seed in self.seed_tmpl.seeds:
            self.stateVec.update({seed: self.T(seed)})

        # for name, content in lemmas:
        #     self.stateVec.update({name: self.T(content)})

        # 4. 总体特征并打分 todo 这里是不是embed太多东西了
        for name, stateVec in self.stateVec.items():
            overall_feature = self.G(self.emb_tla, emb_ce, stateVec)
            predicted_reward = self.P(self.stateVec, overall_feature)
            predicted_reward_list.update({name: predicted_reward})

        # 选择seeds, 生成一条lemma
        action_vector = self.pi(self.stateVec, overall_feature)
        action_distribution, action_raw = self.distributionlize(action_vector, self.stateVec.values())
        new_candidate = sampling(action_distribution, self.seed_tmpl.seeds)

        # 将lemma加入candidates
        self.candidate.update({f"inv_{self.lemma_pointer}": new_candidate[0]})
        self.lemma_pointer += 1

        return self.candidate

    # 第二版
    # def generate_next(self, CE):
    #     self.depth = 1
    #
    #     # 1 初始化
    #     candidate = []
    #     # the lists will be used when punish or prised.
    #     predicted_reward_list = dict()
    #     action_selected_list = []
    #     outputed_list = []
    #     action_or_value = []
    #
    #     # 2 embedding counter example, state
    #     emb_ce = self.E(CE)
    #     tla_ins = self.seed_tmpl.tla_ins
    #     self.emb_tla = self.T.forward_three(tla_ins.init, tla_ins.next, tla_ins.ind)
    #
    #     # 3. 嵌入seeds
    #     seeds = copy.deepcopy(self.seed_tmpl.seeds)
    #     conjuncts = list(seeds)
    #     num_conjuncts = random.randint(config.min_num_conjuncts, config.max_num_conjuncts)
    #
    #     # todo 这里优化之:seed的优先级
    #     c = random.choice(conjuncts)
    #     conjuncts.remove(c)
    #     for seed in self.seed_tmpl.seeds:
    #         self.stateVec.update({seed: self.T(seed)})
    #
    #     # 4. 总体特征并打分 todo 这里是不是embed太多东西了 todo 先cat
    #
    #     # 首先选择seed， 然后随机选取几个seed，为它们选择动作
    #     for name, stateVec in self.stateVec.items():
    #         overall_feature = self.G(self.emb_tla, emb_ce, stateVec)
    #         predicted_reward = self.P(self.stateVec, overall_feature)
    #         predicted_reward_list.update({name: predicted_reward})
    #
    #         # 5. 选择动作，生成lemma
    #         available_acts = get_available_rule()
    #         action_vector = self.pi(self.stateVec, overall_feature)
    #         action_distribution, action_raw = self.distributionlize(action_vector, available_acts)
    #         action_selected = sampling(action_distribution, available_acts)
    #
    #     real_candidate = ""
    #     self.candidate.update({f"lemma_{self.lemma_pointer}": real_candidate})
    #     self.lemma_pointer += 1
    #
    #     # PT = update_PT_rule_selction(PT, left_handle, action_selected)
    #     return self.candidate

    # 处理程序树：在程序树PT的左句柄不为空的情况下，进行以下操作：
    # 获取可用的动作和动作或值的类型。
    # 计算整体特征和预测的奖励。
    # 如果需要选择一个动作，那么就从可用的动作中采样一个动作，并更新程序树。如果深度超过了最大深度，那么就选择最简单的动作。
    # 如果需要选择一个值，那么就从值的分布中采样一个值，并更新程序树。但是请注意，这部分代码目前被注释掉了，所以实际上并不会执行。
    # 更新状态向量和深度：使用新的程序树更新状态向量，并将深度加1。
    # 保存结果：将预测的奖励列表、选择的动作列表、输出的列表、动作或值的类型列表和左句柄列表保存到类的属性中。
    # 返回结果：返回更新后的程序树PT。
    # 第一版
    # def generate_next(self, CE):
    #     self.depth = 0
    #
    #     # 1 初始化
    #     candidate = []
    #     # the lists will be used when punish or prised.
    #     predicted_reward_list = dict()
    #     action_selected_list = []
    #     outputed_list = []
    #     action_or_value = []
    #
    #     # 2 embedding counter example, state
    #     emb_ce = self.E(CE)
    #     tla_ins = self.seed_tmpl.tla_ins
    #     self.emb_tla = self.T.forward_three(tla_ins.init, tla_ins.next, tla_ins.ind)
    #
    #     # 3. 生成lemma, 嵌入lemma todo 看endive思考quants如何添加
    #     lemmas = generate_lemmas(self.seed_tmpl.seeds)
    #     for name, content in lemmas:
    #         self.stateVec.update({name: self.T(content)})
    #
    #     # 4. 总体特征并打分 todo 这里是不是embed太多东西了
    #     for name, stateVec in self.stateVec.items():
    #         overall_feature = self.G(self.emb_tla, emb_ce, stateVec)
    #         predicted_reward = self.P(self.stateVec, overall_feature)
    #         predicted_reward_list.update({name: predicted_reward})
    #
    #     # 选择lemma
    #     result = Lemma2Candidate.select(predicted_reward_list)
    #     real_candidate = []
    #     for name in result.keys():
    #         real_candidate.append(lemmas[name])
    #
    #     # act_or_val, available_acts = AvailableActionSelection(left_handle)
    #     # action_vector = self.pi(self.stateVec, overall_feature)
    #     # if act_or_val == config.SELECT_AN_ACTION:
    #     #     action_distribution, action_raw = self.distributionlize(action_vector, available_acts)
    #     #     action_selected = sampling(action_distribution, available_acts)
    #     #
    #     #     if self.depth >= config.MAX_DEPTH:
    #     #         action_selected = simplestAction(left_handle)
    #     #     action_selected_list.append(action_selected)
    #     #     outputed_list.append(action_raw)
    #     #
    #     #     PT = update_PT_rule_selction(PT, left_handle, action_selected)
    #     #
    #     # else:
    #     #     assert False  # should not be here now
    #
    #     # action_or_value.append(act_or_val)
    #     # left_handle = getLeftHandle(PT)
    #     # self.stateVec = self.T(candidate)
    #     # self.depth += 1
    #     # self.last_predicted_reward_list = predicted_reward_list
    #     # self.last_action_selected_list = action_selected_list
    #     # self.last_outputed_list = outputed_list
    #     # self.last_action_or_value = action_or_value
    #     # self.last_left_handles = left_handles
    #     return real_candidate

    # 设置奖励和伽马值：根据Deg的值设置奖励和伽马值。
    # 如果Deg是 "VERY"，那么奖励是 -10，伽马值是0.1。
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
            self.lemma_pointer -= 1
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
        # for i in range(len(self.last_action_or_value)):
        #     if ShouldStrict(self.last_left_handles[i], Whom):
        #         if self.last_action_or_value[i] == config.SELECT_AN_ACTION:
        #             if SorL == 'STRICT':
        #                 SD = StrictnessDirtribution(self.last_left_handles[i], Whom)
        #             else:
        #                 assert SorL == 'LOOSE'
        #                 SD = LossnessDirtribution(self.last_left_handles[i], Whom)
        #             Loss_strictness = -torch.mm(SD, torch.log_softmax(self.last_outputed_list[i].reshape(1, -1),
        #                                                               1).transpose(0,
        #                                                                            1)) * gama
        #         else:
        #             assert False  # should not be here
        #             # Loss_strictness = F.mse_loss(self.last_outputed_list[i],
        #             #                              torch.tensor([1], dtype=torch.float32)) * gama / 4
        #
        #         strict_loss += Loss_strictness.reshape([1, 1])
        #         counter += 1
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

    def judge_by_time(self, ge_time):
        if ge_time > config.generate_time["very"]:
            self.prise("VERY")
        elif ge_time > config.generate_time["little"]:
            self.prise("LITTLE")
        else:
            self.punish("LOOSE", "LITTLE", "")

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
        paras.update(self.T.get_parameters())
        paras.update(self.G.get_parameters())
        paras.update(self.E.get_parameters())
        paras.update(self.P.get_parameters())
        paras.update(self.pi.get_parameters())
        # paras.update(self.distributionlize.GetParameters())
        # paras.update(self.intValuelzie.GetParameters())
        for parname in paras:
            paras[parname].requires_grad = True

        self.adam = Adam(paras.values(), lr=self.LR)
        self.paras = paras

    # def init_parameters(self):
    #     paradict = initialize_paramethers(self.path2CFile)
    #     for parname in self.paras:
    #         if parname in paradict:  # initialize
    #             assert self.paras[parname].shape == paradict[parname].shape
    #             self.paras[parname].data = paradict[parname].data

    def gpulize(self):
        self.T.cudalize()
        self.G.cudalize()
        self.E.cudalize()
        self.P.cudalize()
        self.pi.cudalize()
        GPUlizeSymbols()
