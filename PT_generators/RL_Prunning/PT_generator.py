from torch.optim import Adam
from PT_generators.RL_Prunning.NNs.NeuralNetwork import *
from PT_generators.RL_Prunning.Template.Seed2Lemma import *
from PT_generators.RL_Prunning.Template.Lemma2Candidate import *
import torch.nn.functional as f


class PT_generator:
    def __init__(self, seed_tmpl, name):
        self.last_predicted_reward_list = None
        self.last_selected_lemma = None
        self.last_distribution_output = None

        self.emb_tla = None
        self.adam = None
        self.paras = None
        self.state_vec = dict()  # key: lemma_name val:lemma_tensor

        self.depth = 0
        self.specname = name
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
        self.distributionlize = construct_distributionlize()  # Distributionlize()

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

    def concat_state_vec(self):
        suma = torch.cat(list(self.state_vec.values()), dim=0)
        concatenated_features = torch.mean(suma, dim=0, keepdim=True)
        return concatenated_features

    def generate_next(self, ce):
        self.depth = 0

        # 1 初始化
        # the lists will be used when punish or prised.
        predicted_reward_list = dict()

        # 2 embedding counter example, state
        emb_ce = self.E(ce)
        tla_ins = self.seed_tmpl.tla_ins
        self.emb_tla = self.T.forward_three(tla_ins.init, tla_ins.next, tla_ins.inv)

        # 3. 嵌入seed
        for seed in self.seed_tmpl.seeds:
            self.state_vec.update({seed: self.T(seed)})

        # for name, content in lemmas:
        #     self.stateVec.update({name: self.T(content)})

        # todo seed2lemma的rule应该和distribution一一对应。
        # 4. 总体特征并打分
        # todo 具体逻辑仍需优化，包括
        # todo 1. statevec 每次更新为lemma和已选择的seed
        # todo 2. overall_feature的具体实现（cfg_emb之类的嵌入函数选择和哪些负责嵌入）

        for name, state_vec in self.state_vec.items():
            overall_feature = self.G(self.emb_tla, emb_ce, state_vec)
            predicted_reward = self.P(state_vec, overall_feature)
            predicted_reward_list.update({name: predicted_reward})

        # 选择seeds, 生成一条lemma
        action_vector = self.pi(self.concat_state_vec(), overall_feature)
        action_distribution, action_raw = self.distributionlize(action_vector, list(self.state_vec.values()))

        seeds_num = random.randint(3, 7)
        new_candidate, raw_lemmas = sampling(action_distribution, self.seed_tmpl.seeds, seeds_num)

        # 将lemma加入candidates
        self.candidate.update({f"inv_{self.lemma_pointer}": new_candidate[0]})
        self.lemma_pointer += 1

        self.last_predicted_reward_list = predicted_reward_list
        self.last_selected_lemma = raw_lemmas
        self.last_distribution_output = action_raw

        lemmas = Lemma2Candidate.add_quant(f"inv_{self.lemma_pointer - 1}", new_candidate[0], self.seed_tmpl.quants)

        return self.candidate, lemmas, self.lemma_pointer - 1

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
    def punish(self, s_or_l, deg, whom):
        if deg == "VERY":
            reward = -10
            gama = 0.1
            self.lemma_pointer -= 1
        elif deg == "MEDIUM":
            reward = -5
            gama = 0.05
        elif deg == "LITTLE":
            reward = -1
            gama = 0.01
        else:
            reward = 0
            gama = 0.01

        strict_loss = tensor([[0]], dtype=torch.float32)
        counter = 0
        length = len(self.last_selected_lemma)
        print(self.last_selected_lemma)
        for i in range(length):
            if s_or_l == 'STRICT':  # strict倾向于选择更长的子句
                sd = strictness_distribution(self.seed_tmpl.seeds, self.last_selected_lemma[i], length)
            else:  # loose倾向于选择更短的子句
                sd = looseness_distribution(self.seed_tmpl.seeds, self.last_selected_lemma[i], length)
            loss_strictness = -torch.mm(sd, torch.log_softmax(
                self.last_distribution_output.reshape(1, -1), 1).transpose(0, 1)) * gama
            strict_loss += loss_strictness.reshape([1, 1])
            counter += 1
        if counter != 0:
            strict_loss /= counter
        a_loss = self.a_loss(reward)
        self.learn_step((a_loss + strict_loss))

    # 计算奖励列表：根据final_reward和discounter计算每一步的奖励，并将结果存储在reward_list中。
    # 计算预测损失：对于self.last_predicted_reward_list中的每一个元素，计算预测的奖励和上一步的预测奖励，然后根据动作的类型计算损失。如果这个动作是选择一个动作，那么就计算交叉熵损失。如果这个动作是选择一个值，那么就计算均方误差损失（但是请注意，这部分代码目前被注释掉了，所以实际上并不会执行）。然后将损失乘以奖励和上一步的预测奖励的差值，并累加到总的预测损失上。
    # 计算平均预测损失：将总的预测损失除以奖励列表的长度，得到平均预测损失。
    # 计算均方误差损失：使用F.mse_loss函数计算奖励列表和预测奖励列表之间的均方误差损失。
    # 返回总损失：将预测损失和均方误差损失相加，得到总损失，并返回。

    def a_loss(self, final_reward):
        discounter = 0.95
        reward_list = []
        for i in range(len(self.last_predicted_reward_list)):
            reward_list.append(final_reward * discounter ** i)
        reward_list = reward_list[::-1]
        p_loss = 0
        for i in range(len(self.last_selected_lemma)):
            r_i = reward_list[i]
            if i == 0:
                pr_i_1 = tensor([[0]], dtype=torch.float32)
            else:
                pr_i_1 = tensor([reward_list[i - 1]], dtype=torch.float32)

            losser = f.cross_entropy(self.last_distribution_output.reshape(1, -1),
                                     get_action_index(self.last_selected_lemma[i], self.seed_tmpl.seeds))

            if torch.cuda.is_available():
                p_loss += (tensor(r_i, dtype=torch.float32) - pr_i_1).cuda() * losser.reshape([1, 1])
            else:
                p_loss += (tensor(r_i, dtype=torch.float32) - pr_i_1) * losser.reshape([1, 1])
        p_loss = p_loss / len(reward_list)
        if torch.cuda.is_available():
            mse_loss = f.mse_loss(tensor([reward_list], dtype=torch.float32).cuda(),
                                  torch.cat(list(self.last_predicted_reward_list.values()), 1)).reshape([1, 1])
        else:
            mse_loss = f.mse_loss(tensor([reward_list], dtype=torch.float32),
                                  torch.cat(list(self.last_predicted_reward_list.values()), 1)).reshape([1, 1])
        # print("mse_loss", mse_loss)
        # print("p_loss", p_loss)
        return p_loss + mse_loss

    def judge_by_time(self, ge_time):
        if ge_time > config.generate_time["very"]:
            self.prise("VERY")
        elif ge_time > config.generate_time["little"]:
            self.prise("LITTLE")
        else:
            self.punish("LOOSE", "LITTLE", "")

    def prise(self, deg):
        if deg == "VERY":
            reward = 10
        elif deg == "LITTLE":
            reward = 1
        else:
            reward = 0
        a_loss = self.a_loss(reward)
        self.learn_step(a_loss)

    def learn_step(self, loss):
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
        for par_name in paras:
            paras[par_name].requires_grad = True

        self.adam = Adam(paras.values(), lr=self.LR)
        self.paras = paras

    def gpulize(self):
        self.T.cudalize()
        self.G.cudalize()
        self.E.cudalize()
        self.P.cudalize()
        self.pi.cudalize()
        GPUlizeSymbols()
