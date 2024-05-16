from Utilities.Logging import log
import torch
from torch import tensor
import numpy as np
from sympy import *
from itertools import product
from PT_generators.RL_Prunning.Conifg import config


def generate_combinations(expressions):
    combinations = []
    n = len(expressions)
    for selection in product([1, -1, 0], repeat=n):
        combination = []
        for i in range(n):
            if selection[i] == 1:
                combination.append(expressions[i])
            elif selection[i] == -1:
                combination.append(f'~({expressions[i]})')
        combinations.append(combination)
    return combinations


class seed2Lemma:
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

    # RULE = {
    #     "all": [op_and, op_or, op_neg]
    # }

    DIST = []


# def get_available_rule():
#     return RULE["all"]


def get_seed_index(last_seed, seed_list):
    for i, action in enumerate(seed_list):
        if str(action) in str(last_seed):
            if torch.cuda.is_available():
                return tensor([i]).cuda()
            else:
                return tensor([i])
    assert False  # should not be here


def generate_lemmas(depth, seeds: list):
    """ Generate 'num_invs' random invariants with the specified number of conjuncts. """
    # Pick some random conjunct.
    # OR and negations should be functionally complete
    symb_neg_op = "~"
    ops = ["\\/"]
    and_op = "/\\"
    neg_op = "~"

    # Assign a numeric id to each predicate.
    seed_id = {p: k for (k, p) in enumerate(seeds)}

    invs = dict()
    # invs_sym = []
    # invs_sym_strs = []

    # choose = list()

    # 示例用法
    combinations = generate_combinations(seeds)
    for i, combination in enumerate(combinations):
        invs[f"inv_{depth}_{i}"] =  ' \\/ '.join(combination)

    # for inv_id in range(num_invs):
    #     conjuncts = list(seeds)
    #     # conjuncts = list(map(str, range(len(preds))))
    #     num_conjuncts = random.randint(min_num_conjuncts, max_num_conjuncts)
    #     num_conjuncts = min(num_conjuncts, len(seeds))
    #
    #     # Select first atomic term of overall predicate.
    #     c = random.choice(conjuncts)
    #     conjuncts.remove(c)
    #
    #     # Optionally negate it.
    #     negate = random.choice([True, False])
    #     (n, fn) = (neg_op, symb_neg_op) if negate else ("", "")
    #
    #     inv = n + "(" + c + ")"
    #     choose.append(inv)
    #     pred_id_var = f"x_{str(seed_id[c]).zfill(3)}"
    #     symb_inv_str = fn + "(" + pred_id_var + ")"
    #
    #     for i in range(1, num_conjuncts):
    #         c = random.choice(conjuncts)
    #         conjuncts.remove(c)
    #         op = ""
    #         fop = "|"
    #         if i < num_conjuncts:
    #             op = random.choice(ops)
    #         negate = random.choice([True, False])
    #         (n, fn) = (neg_op, symb_neg_op) if negate else ("", "")
    #         new_term = n + "(" + c + ")"
    #         choose.append(new_term)
    #
    #         # Sort invariant terms to produce more consistent output regardless of random seed.
    #         new_inv_args = [new_term, inv]
    #         new_inv_args = sorted(new_inv_args)
    #         inv = new_inv_args[0] + " " + op + " (" + new_inv_args[1] + ")"
    #
    #         # inv  =  n + "(" + c + ")" + " " + op + " (" + inv +")"
    #
    #         # # Symbolic version of the predicate. Used for quickly
    #         # # detecting logically equivalent predicate forms.
    #         # pred_id_var = f"x_{str(seed_id[c]).zfill(3)}"
    #         # symb_inv_str = fn + "(" + pred_id_var + ")" + " " + fop + " (" + symb_inv_str + ")"
    #
    #     if inv not in invs:
    #         invs.update({inv_id: inv})
    #         # invs_sym.append(pyeda.inter.expr(symb_inv_str))
    #         # print(type(invs_sym[-1]))
    #         invs_sym_strs.append(symb_inv_str)

    log.info(f"generate {len(invs)} invs.")

    # # Do CNF based equivalence reduction.
    # invs = symb_equivalence_reduction(invs, invs_sym)
    # logging.info(f"number of invs post CNF based equivalence reduction: {len(invs)}")

    # if len(quant_vars):
    # invs = pred_symmetry_reduction(invs, quant_vars)

    # return invs_sym
    # return invs_sym_strs
    # return set(map(str, invs_sym))
    # return {"raw_invs": set(invs), "pred_invs": invs_sym_strs}
    return invs, combinations


# def init_dict(seed_list):
#     DIST = [1 / len(seed_list)] * len(seed_list)


def strictness_distribution(seed_list, seed, length):
    distri_dict = [1 / len(seed_list)] * len(seed_list)
    res = torch.ones(len(seed_list), 1, dtype=torch.float32)
    for i, every_seed in enumerate(seed_list):
        if every_seed == seed:
            res[i, 0] = res[i, 0] * distri_dict[i]
            distri_dict[i] += 1 / len(seed_list)

    normalization(distri_dict)
    if torch.cuda.is_available():
        res = res.cuda()
    return res


def looseness_distribution(seed_list, seeds_selected):
    distri_dict = [1 / len(seed_list)] * len(seed_list)
    res = torch.ones(len(seed_list), 1, dtype=torch.float32)
    for i, every_seed in enumerate(seed_list):
        res[i, 0] = distri_dict[i]
        for j, seed in enumerate(seeds_selected):
            if every_seed == seed:
                res[i, 0] = 1
                break

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
    # print("lister ",lister)
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
