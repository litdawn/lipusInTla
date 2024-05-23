import random
from PT_generators.RL_Prunning.Template.Seed2Lemma import sampling
from PT_generators.RL_Prunning.Conifg import config

class PT_generator:
    def __init__(self, seed_tmpl, name):
        self.already_generate = set()
        self.seed_tmpl = seed_tmpl
        self.spec_name = name
        self.lemma_pointer = 0
        self.candidate = dict()
        self.seeds = seed_tmpl.seeds
        self.depth = 0
        self.last_generate_invs = None
        self.reward_list = []

    # def init_candidate(self):
    #     self.candidate.clear()
    #     if config.use_self_generate:
    #         self.candidate.update({"Safety": self.seed_tmpl.tla_ins.inv})
    #         self.candidate.update({"Typeok": self.seed_tmpl.tla_ins.type_ok})
    #     else:
    #         self.candidate.update({"Safety": self.seed_tmpl.safety})
    #         self.candidate.update({"Typeok": self.seed_tmpl.typeok})

    def generate_next(self, cti):
        seeds_num = random.randint(2, 3)
        new_candidate, raw_lemmas = sampling([], self.seeds,self.depth,True)
        while len(new_candidate)==0:
            new_candidate, raw_lemmas = sampling([], self.seeds, self.depth,True)
        lemmas = {}
        for name, inv in new_candidate.items():
            lemmas.update({name: f"{self.seed_tmpl.quant_inv} {inv}"})
        # print(lemmas[0])
        self.depth += 1
        self.last_generate_invs = lemmas
        return self.candidate, lemmas

    def update_candidate(self, names: list):
        for name in names:
            self.candidate.update({name: self.last_generate_invs[name]})

    def punish(self, s_or_l, deg):
        pass

    def prise(self, deg, successes):
        self.update_candidate(list(successes.keys()))
        pass


# if __name__ == "__main__":
#     list1 = [1,2]
#     list2 = [2,1]
#     all_list = set()
#     all_list.update(set(list1))
#
#     print(set(list2) in all_list)
