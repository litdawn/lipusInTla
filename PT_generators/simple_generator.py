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
        self.init_candidate()

    def init_candidate(self):
        self.candidate.clear()
        if config.use_self_generate:
            self.candidate.update({"Safety": self.seed_tmpl.tla_ins.inv})
            self.candidate.update({"Typeok": self.seed_tmpl.tla_ins.type_ok})
        else:
            self.candidate.update({"Safety": self.seed_tmpl.safety})
            self.candidate.update({"Typeok": self.seed_tmpl.typeok})

    def generate_next(self, cti):
        # seeds_num = random.randint(2, 4)
        seeds_num = 2
        new_candidate, raw_lemmas = sampling([], self.seeds, seeds_num,True)
        while self.if_already_generate(raw_lemmas):
            new_candidate, raw_lemmas = sampling([], self.seeds, seeds_num,True)

        self.candidate.update({f"inv_{self.lemma_pointer}": new_candidate[0]})
        self.lemma_pointer += 1
        lemmas = [f"inv_{self.lemma_pointer - 1} ==  {self.seed_tmpl.quant_inv} {new_candidate[0]}"]

        self.cache_inv(raw_lemmas)
        # print(lemmas[0])
        return self.candidate, lemmas, self.lemma_pointer - 1

    def cache_inv(self, inv):
        self.already_generate.add(tuple(inv))
        # self.all_checked_invs = self.all_checked_invs.union(set(map(quant_inv_fn, list(invs))))
        return

    def punish(self, s_or_l, deg):
        if deg == "VERY":
            self.lemma_pointer -=1
        pass

    def prise(self, deg):
        pass

    def if_already_generate(self, inv):
        if tuple(inv) in self.already_generate:
            return True
        return False

if __name__ == "__main__":
    list1 = [1,2]
    list2 = [2,1]
    all_list = set()
    all_list.update(set(list1))

    print(set(list2) in all_list)
