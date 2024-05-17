from torch.optim import Adam
from PT_generators.RL_Prunning.NNs.NeuralNetwork import *
from PT_generators.RL_Prunning.Template.Seed2Lemma import *
from PT_generators.RL_Prunning.Template.Lemma2Candidate import *
import torch.nn.functional as f


class PT_generator:
    def __init__(self, seed_tmpl, name):
        self.loss_list = []
        self.last_predicted_reward_list = None
        self.reward_list = []
        self.last_selected_lemma = None
        self.last_distribution_output = None
        # self.already_generate = set()

        self.emb_tla = None
        self.adam = None
        self.paras = None
        self.state_vec = dict()  # key: lemma_name val:lemma_tensor

        self.depth = 0
        self.specname = name
        self.LR = config.LearningRate
        # Step1. Parse the inputs
        self.seed_tmpl = seed_tmpl

        init_symbolEmbeddings(seed_tmpl)

        # Step2. Construct the NNs and Load the parameters
        # 在 NNs.NeuralNetwork 里
        self.T = constructT(seed_tmpl)  # treeLSTM
        self.G = constructG()  # Overall_Embedding
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
        # self.lemma_pointer = 0
        # self.init_candidate()

    # def init_candidate(self):
    #     self.candidate.clear()
    #     if config.use_self_generate:
    #         self.candidate.update({"Safety": self.seed_tmpl.tla_ins.inv})
    #         self.candidate.update({"Typeok": self.seed_tmpl.tla_ins.type_ok})
    #     else:
    #         self.candidate.update({"Safety": self.seed_tmpl.safety})
    #         self.candidate.update({"Typeok": self.seed_tmpl.typeok})

    def concat_state_vec(self):
        suma = torch.cat(list(self.state_vec.values()), dim=0)
        concatenated_features = torch.mean(suma, dim=0, keepdim=True)
        return concatenated_features

    def generate_next(self, cti):

        # 1 初始化
        # the lists will be used when punish or prised.
        predicted_reward_list = dict()

        # 2 embedding counter example, state
        emb_cti = self.E(cti)
        tla_ins = self.seed_tmpl.tla_ins
        # if config.use_self_generate:
        self.emb_tla = self.T.forward_three(tla_ins.init, tla_ins.next, tla_ins.inv)
        # else:
        #     self.emb_tla = self.T.forward_three()
        # 3. 嵌入seed
        for seed in self.seed_tmpl.seeds:
            self.state_vec.update({seed: self.T(seed, "seed")})

        # todo seed2lemma的rule应该和distribution一一对应。
        # 4. 总体特征并打分
        # todo 具体逻辑仍需优化，包括
        # todo 1. statevec 每次更新为lemma和已选择的seed
        # todo 2. overall_feature的具体实现（cfg_emb之类的嵌入函数选择和哪些负责嵌入）

        for name, state_vec in self.state_vec.items():
            overall_feature = self.G(self.emb_tla, emb_cti, state_vec)
            predicted_reward = self.P(state_vec, overall_feature)
            predicted_reward_list.update({name: predicted_reward})

        # 选择seeds, 生成一条lemma
        action_vector = self.pi(self.concat_state_vec(), overall_feature)
        action_distribution, action_raw = self.distributionlize(action_vector, list(self.state_vec.values()))

        new_candidate, raw_lemmas = sampling(action_distribution, self.seed_tmpl.seeds, self.depth)
        self.depth += 1


        self.last_predicted_reward_list = predicted_reward_list
        self.last_selected_lemma = raw_lemmas
        self.last_distribution_output = action_raw
        if config.use_self_generate:
            lemmas = Lemma2Candidate.add_quant(f"inv_{self.lemma_pointer - 1}", new_candidate[0], self.seed_tmpl.quants)
        else:
            lemmas = []
            for name, inv in new_candidate.items():
                lemmas = [f"{name} ==  {self.seed_tmpl.quant_inv} {inv}"]

        return self.candidate, lemmas

    def decide_little_prise(self, successes):
        reward = {}
        for name, cti_num in successes.items():
            reward.update({name: cti_num * cti_num})
        return reward

    def decide_very_prise(self, successes):
        return 10

    def prise(self, deg, successes: dict):
        self.candidate = self.candidate.update(successes)
        if deg == "VERY":
            reward = self.decide_very_prise(successes)
        elif deg == "LITTLE":
            reward = self.decide_little_prise(successes)
        else:
            reward = {name: 0 for name in successes.keys()}

        self.reward_list.extend(reward.values())
        for name, val in reward:
            a_loss = self.a_loss(val, name)
            self.learn_step(a_loss)

    def decide_very_punish(self, failures:dict):
        reward = dict()
        for name, sth in failures.items():
            reward.update({name:(-10, 0.1)})
        return reward

    def decide_little_punish(self, failures):
        reward = dict()
        for name, sth in failures.items():
            reward.update({name:(-1, 0.2)})
        return reward

    def punish(self, deg, failures: dict):
        if deg == "VERY":
            reward =  self.decide_very_punish(failures)
        elif deg == "MEDIUM":
            reward = {name: (-5, 0.05) for name in failures.keys()}
        elif deg == "LITTLE":
            reward = self.decide_little_punish(failures)
        else:
            reward = {name: (0, 0.01) for name in failures.keys()}
        self.reward_list.extend([a[0] for a in reward.values()])

        for name, val in reward.items():
            sd = strictness_distribution(self.seed_tmpl.seeds, self.last_selected_lemma[name])
            loss_strictness = -torch.mm(torch.log_softmax(
                self.last_distribution_output.reshape(1, -1), 1), sd) * val[1]
            strict_loss = loss_strictness.reshape([1, 1]) / len(self.last_selected_lemma[name])
            a_loss = self.a_loss(val[0], name)
            self.learn_step((a_loss + strict_loss))

    # 计算奖励列表：根据final_reward和discounter计算每一步的奖励，并将结果存储在reward_list中。
    # 计算预测损失：对于self.last_predicted_reward_list中的每一个元素，计算预测的奖励和上一步的预测奖励，然后根据动作的类型计算损失。如果这个动作是选择一个动作，那么就计算交叉熵损失。如果这个动作是选择一个值，那么就计算均方误差损失（但是请注意，这部分代码目前被注释掉了，所以实际上并不会执行）。然后将损失乘以奖励和上一步的预测奖励的差值，并累加到总的预测损失上。
    # 计算平均预测损失：将总的预测损失除以奖励列表的长度，得到平均预测损失。
    # 计算均方误差损失：使用F.mse_loss函数计算奖励列表和预测奖励列表之间的均方误差损失。
    # 返回总损失：将预测损失和均方误差损失相加，得到总损失，并返回。

    def a_loss(self, final_reward, lemma_name):
        discounter = 0.95
        reward_list = []
        for i in range(len(self.last_predicted_reward_list)):
            reward_list.append(final_reward * discounter ** i)
        reward_list = reward_list[::-1]
        p_loss = 0
        for i in range(min(len(self.last_selected_lemma[lemma_name]), len(reward_list))):
            r_i = reward_list[i]
            if i == 0:
                pr_i_1 = tensor([[0]], dtype=torch.float32)
            else:
                pr_i_1 = tensor([reward_list[i - 1]], dtype=torch.float32)

            losser = f.cross_entropy(self.last_distribution_output.reshape(1, -1),
                                     get_seed_index(self.last_selected_lemma[lemma_name][i], self.seed_tmpl.seeds))

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

    # def judge_by_time(self, ge_time):
    #     if ge_time > config.generate_time["very"]:
    #         self.prise("VERY")
    #     elif ge_time > config.generate_time["little"]:
    #         self.prise("LITTLE")
    #     else:
    #         self.punish("LOOSE", "LITTLE")

    def learn_step(self, loss):
        # if torch.cuda.is_available():
        #     loss = loss.cuda()
        self.loss_list.append(loss[0][0].item())
        self.adam.zero_grad()
        # print(loss)
        loss.backward(retain_graph=True)
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
