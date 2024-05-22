import random

import torch
from torch.optim import Adam
from PT_generators.RL_Prunning.NNs.NeuralNetwork import *
from PT_generators.RL_Prunning.Template.Seed2Lemma import *
from PT_generators.RL_Prunning.Template.Lemma2Candidate import *
import torch.nn.functional as f
from torch import tensor
from memory_profiler import profile


# torch.autograd.set_detect_anomaly(True)


class PT_generator:
    def __init__(self, seed_tmpl, name):
        self.loss_list = []
        self.last_predicted_reward_list = None
        self.reward_list = []
        self.last_selected_lemma = None
        self.last_generate_invs = None
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

        self.candidate = {}
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

    # @profile()
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
        while len(new_candidate) == 0:
            new_candidate, raw_lemmas = sampling(action_distribution, self.seed_tmpl.seeds, self.depth)

        self.depth += 1

        self.last_predicted_reward_list = predicted_reward_list
        self.last_selected_lemma = raw_lemmas

        self.last_distribution_output = action_raw
        if config.use_self_generate:
            lemmas = Lemma2Candidate.add_quant(f"inv_{self.lemma_pointer - 1}", new_candidate[0], self.seed_tmpl.quants)
        else:
            lemmas = {}
            for name, inv in new_candidate.items():
                lemmas.update({name: f"{self.seed_tmpl.quant_inv} {inv}"})
        self.last_generate_invs = lemmas
        return self.candidate, lemmas

    def update_candidate(self, names: list):
        for name in names:
            self.candidate.update({name: self.last_generate_invs[name]})

    def prise(self, deg, successes: dict):
        self.update_candidate(list(successes.keys()))

        def _normalization(target: dict, min_interval, max_interval):
            min_val = 0 if len(target) == 0 else min(target.values())
            max_val = 0 if len(target) == 0 else max(target.values())
            range_val = max_val - min_val if max_val - min_val > 0 else 1
            normalized_dict = {}
            for key, _val in target.items():
                normalized_val = ((_val - min_val) / range_val) * (max_interval - min_interval) + min_interval
                normalized_dict[key] = normalized_val
            return normalized_dict

        def decide_little_prise(_successes):
            return _normalization(_successes, 2, 100)

        def decide_very_prise():
            return 101

        if deg == "VERY":
            reward = decide_very_prise()
        elif deg == "LITTLE":
            reward = decide_little_prise(successes)
        else:
            reward = {name: 0 for name in successes.keys()}

        self.reward_list.extend(reward.values())

        if torch.cuda.is_available():
            a_loss = tensor([[0]], dtype=torch.float32).cuda()
        else:
            a_loss = tensor([[0]], dtype=torch.float32)
        for name, val in reward.items():
            a_loss += self.a_loss(val, name)
        self.learn_step(a_loss)

    # @profile()
    def punish(self, deg, failures: dict):
        def _normalization(target: dict, min_interval, max_interval):
            min_val = min(target.values())
            max_val = max(target.values())
            range_val = max_val - min_val if max_val - min_val > 0 else 1
            normalized_dict = {}
            for key, _val in target.items():
                normalized_val = (((_val[0] - min_val) / range_val) * (max_interval - min_interval) + min_interval, 0.1)
                normalized_dict[key] = normalized_val
            return normalized_dict

        def decide_very_punish(_failures: dict):
            return _normalization(_failures, -100, -5)

        def decide_little_punish(_failures: dict):
            _reward = dict()
            for name, sth in _failures.items():
                _reward.update({name: (-1, 0.2)})
            return _reward

        if deg == "VERY":
            reward = decide_very_punish(failures)
        elif deg == "MEDIUM":
            reward = {name: (-5, 0.05) for name in failures.keys()}
        elif deg == "LITTLE":
            reward = decide_little_punish(failures)
        else:
            reward = {name: (0, 0.01) for name in failures.keys()}
        self.reward_list.extend([a[0] for a in reward.values()])

        if torch.cuda.is_available():
            strict_loss = tensor([[0]], dtype=torch.float32).cuda()
            a_loss = tensor([[0]], dtype=torch.float32).cuda()
        else:
            strict_loss = tensor([[0]], dtype=torch.float32)
            a_loss = tensor([[0]], dtype=torch.float32)
        for name, val in reward.items():
            sd = strictness_distribution(self.seed_tmpl.seeds, self.last_selected_lemma[name])
            loss_strictness = -torch.mm(torch.log_softmax(
                self.last_distribution_output.reshape(1, -1), 1), sd) * val[1]
            strict_loss += loss_strictness.reshape([1, 1]) / len(self.last_selected_lemma[name])
            a_loss += self.a_loss(val[0], name) / len(self.last_selected_lemma[name])
        self.learn_step((a_loss + strict_loss))

    # @profile()
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
        return p_loss + mse_loss

    # @profile()
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
