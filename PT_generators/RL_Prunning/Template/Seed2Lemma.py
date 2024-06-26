import logging
import random
import torch
from torch import tensor
import numpy as np
from PT_generators.RL_Prunning.NNs.SymbolEmbeddings import SymbolEmbeddings
from PT_generators.RL_Prunning.Conifg import config


def op_and(*args):
    return f"({args[0]})/\\({args[1]})"


def op_and_str():
    return "and"


def op_or(*args):
    return f"({args[0]})\\/({args[1]})"


def op_or_str():
    return "or"


def op_neg(*args):
    return f"(~ {args[0]})"


def op_neg_str():
    return "neg"


op_and.__str__ = op_and_str
op_or.__str__ = op_or_str
op_neg_str.__str__ = op_neg_str

# RULE = {
#     "all": [op_and, op_or, op_neg]
# }

RULE = {
    "all": [2, 3, 4, 5, 6, 7]
    # todo 应该是in/ subseteq/ = /[]/ ()几类
}


# RULE = {
#     "and": op_and,
#     "or": op_or,
#     "neg": op_neg
#
# }

def get_available_rule():
    return RULE["all"]


def get_action_index(last_seed, seed_list):
    for i, action in enumerate(seed_list):
        if str(action) == str(last_seed):
            if torch.cuda.is_available():
                return tensor([i]).cuda()
            else:
                return tensor([i])

    assert False  # should not be here


def generate_lemmas(seeds: list, min_num_conjuncts=2, max_num_conjuncts=5, num_invs=1):
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
    invs_sym = []
    invs_sym_strs = []

    for inv_id in range(num_invs):
        conjuncts = list(seeds)
        # conjuncts = list(map(str, range(len(preds))))
        num_conjuncts = random.randint(min_num_conjuncts, max_num_conjuncts)

        # Select first atomic term of overall predicate.
        c = random.choice(conjuncts)
        conjuncts.remove(c)

        # Optionally negate it.
        negate = random.choice([True, False])
        (n, fn) = (neg_op, symb_neg_op) if negate else ("", "")

        inv = n + "(" + c + ")"
        pred_id_var = f"x_{str(seed_id[c]).zfill(3)}"
        symb_inv_str = fn + "(" + pred_id_var + ")"

        for i in range(1, num_conjuncts):
            c = random.choice(conjuncts)
            conjuncts.remove(c)
            op = ""
            fop = "|"
            if i < num_conjuncts:
                op = random.choice(ops)
            negate = random.choice([True, False])
            (n, fn) = (neg_op, symb_neg_op) if negate else ("", "")
            new_term = n + "(" + c + ")"

            # Sort invariant terms to produce more consistent output regardless of random seed.
            new_inv_args = [new_term, inv]
            new_inv_args = sorted(new_inv_args)
            inv = new_inv_args[0] + " " + op + " (" + new_inv_args[1] + ")"

            # inv  =  n + "(" + c + ")" + " " + op + " (" + inv +")"

            # # Symbolic version of the predicate. Used for quickly
            # # detecting logically equivalent predicate forms.
            # pred_id_var = f"x_{str(seed_id[c]).zfill(3)}"
            # symb_inv_str = fn + "(" + pred_id_var + ")" + " " + fop + " (" + symb_inv_str + ")"

        if inv not in invs:
            invs.update({inv_id: inv})
            # invs_sym.append(pyeda.inter.expr(symb_inv_str))
            # print(type(invs_sym[-1]))
            invs_sym_strs.append(symb_inv_str)

    logging.info(f"number of invs: {len(invs)}")

    # # Do CNF based equivalence reduction.
    # invs = symb_equivalence_reduction(invs, invs_sym)
    # logging.info(f"number of invs post CNF based equivalence reduction: {len(invs)}")

    # if len(quant_vars):
    # invs = pred_symmetry_reduction(invs, quant_vars)
    logging.info(f"number of post symmetry invs: {len(invs)}")

    # return invs_sym
    # return invs_sym_strs
    # return set(map(str, invs_sym))
    # return {"raw_invs": set(invs), "pred_invs": invs_sym_strs}
    return invs


def strictness_distribution(seed_list, seed, length):
    distri_dict = {
        "all": [0.05, 0.05, 0.2, 0.3, 0.3, 0.1]
    }
    gamma = distri_dict["all"][length - 2]
    res = torch.ones(len(seed_list), 1, dtype=torch.float32)
    for i, every_seed in enumerate(seed_list.keys()):
        if every_seed == seed:
            res[i, 0] = res[i, 0] * gamma

    if torch.cuda.is_available():
        res = res.cuda()
    return res


def looseness_distribution(seed_list, seed, length):
    distri_dict = {
        "all": [0.1, 0.2, 0.3, 0.3, 0.05, 0.05]
    }
    print(length)
    gamma = distri_dict["all"][length - 2]
    res = torch.ones( 1, len(seed_list), dtype=torch.float32)
    for i, every_seed in enumerate(seed_list):
        if every_seed == seed:
            res[0, i] = res[0, i] * gamma

    if torch.cuda.is_available():
        res = res.cuda()
    return res


def normalization(dist):
    lister = [float(x) for x in list(dist[0])]
    sumer = sum(lister)
    lister = [x / sumer for x in lister]
    # print("lister ",lister)
    return lister


def sampling(action_distribution, sample_list: list, seeds_num=5):
    seeds_selected = []
    # print(action_distribution.shape)
    # if best:
    #     xx = np.asarray(action_distribution[0])
    #     top_idx = xx.argsort()[-1:(-seeds_num - 1):-1]
    #     seeds_selected = sample_list[top_idx]
    # else:
    try:
        # print(sample_list)
        seeds_selected = np.random.choice(sample_list, size=seeds_num, replace=False,
                                          p=normalization(action_distribution))
        # print(seeds_selected)
    except Exception as e:
        print("shit", e)
        raise e
    return generate_lemmas(seeds_selected), seeds_selected
