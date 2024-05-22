from Utilities.Logging import log
import logging
import torch
# from torch import tensor
import numpy as np
from sympy import *
from itertools import product
from PT_generators.RL_Prunning.Conifg import config


class Seed2Lemma:
    seed_tuples = set()
    tmp_tuples = set()

    def op_and(self, *args):
        return f"({args[0]})/\\({args[1]})"

    def op_and_str(self):
        return "and"

    def op_or(self, *args):
        return f"({args[0]})\\/({args[1]})"

    def op_or_str(self):
        return "or"

    def op_neg(self, *args):
        return f"(~ {args[0]})"

    def op_neg_str(self):
        return "neg"

    op_and.__str__ = op_and_str
    op_or.__str__ = op_or_str
    op_neg_str.__str__ = op_neg_str

    DIST = []


seed2lemma = Seed2Lemma()


def get_seed_index(last_seed, seed_list, is_tensor=True):
    for i, action in enumerate(seed_list):
        if str(action) in str(last_seed):
            if torch.cuda.is_available():
                return torch.tensor([i]).cuda() if is_tensor else i
            else:
                return torch.tensor([i]) if is_tensor else i
    assert False  # should not be here


def generate_combinations(expressions):
    combinations = []
    n = len(expressions)
    for selection in product([1, -1, 0], repeat=n):
        combination = []
        selected = []
        for i in range(n):
            if selection[i] == 1:
                combination.append(f'({expressions[i]})')
                selected.append(expressions[i])
            elif selection[i] == -1:
                combination.append(f'~({expressions[i]})')
                selected.append(expressions[i])
        selected.sort()
        combination.sort()
        if tuple(selected) not in seed2lemma.seed_tuples and len(selected) > 0:
            seed2lemma.tmp_tuples.add(tuple(selected))
            combinations.append(combination)
    seed2lemma.seed_tuples = seed2lemma.seed_tuples.union(seed2lemma.tmp_tuples)
    return combinations


def generate_lemmas(depth: int, seeds: list):
    """ Generate 'num_invs' random invariants with the specified number of conjuncts. """
    invs = dict()
    combinations = generate_combinations(seeds)
    combinations_with_name = {}
    for i, combination in enumerate(combinations):
        invs.update({f"Inv_{depth}_{i}": ' \\/ '.join(combination)})
        combinations_with_name.update({f"Inv_{depth}_{i}": combination})

    logging.info(f"generate {len(invs)} invs.")
    return invs, combinations_with_name


# def init_dict(seed_list):
#     DIST = [1 / len(seed_list)] * len(seed_list)


def strictness_distribution(seed_list, seeds_selected):
    distri_dict = [1 / len(seed_list)] * len(seed_list)
    res = torch.ones(len(seed_list), 1, dtype=torch.float32)
    for i, every_seed in enumerate(seeds_selected):
        begin = every_seed.find("(")
        end = len(every_seed) - 1
        every_seed = every_seed[begin + 1:end]
        res[get_seed_index(every_seed, seed_list, False), 0] *= distri_dict[i]
    if torch.cuda.is_available():
        res = res.cuda()
    return res


def normalization(dist):
    if type(dist) is list:
        lister = [float(x) for x in dist]
    else:
        lister = [float(x) for x in list(dist[0])]
    sumer = sum(lister)
    lister = [x / sumer for x in lister]
    return lister


def sampling(action_distribution, sample_list: list, i, pure_random=False):
    seeds_selected = []
    # print(action_distribution.shape)
    # if best:
    #     xx = np.asarray(action_distribution[0])
    #     top_idx = xx.argsort()[-1:(-seeds_num - 1):-1]
    #     seeds_selected = sample_list[top_idx]
    # else:
    if pure_random:
        seeds_selected = np.random.choice(sample_list, size=config.seed_num, replace=False)
    else:
        try:
            seeds_selected = np.random.choice(sample_list, size=min(config.seed_num, len(sample_list)), replace=False,
                                              p=normalization(action_distribution))
        except Exception as e:
            print("错误：请检查config文件中的seed是否重复")
            raise e

    invs, choose = generate_lemmas(i, seeds_selected)
    return invs, choose
