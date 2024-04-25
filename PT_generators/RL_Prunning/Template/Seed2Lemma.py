import logging
import random


def generate_lemmas( seeds: list, min_num_conjuncts=2, max_num_conjuncts=4, num_invs=150):
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
    invs_symb = []
    invs_symb_strs = []

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
            invs.update({inv_id: f"inv_{inv_id} == {inv}"})
            # invs_symb.append(pyeda.inter.expr(symb_inv_str))
            # print(type(invs_symb[-1]))
            invs_symb_strs.append(symb_inv_str)

    logging.info(f"number of invs: {len(invs)}")

    # # Do CNF based equivalence reduction.
    # invs = symb_equivalence_reduction(invs, invs_symb)
    # logging.info(f"number of invs post CNF based equivalence reduction: {len(invs)}")

    # if len(quant_vars):
    # invs = pred_symmetry_reduction(invs, quant_vars)
    logging.info(f"number of post symmetry invs: {len(invs)}")

    # return invs_symb
    # return invs_symb_strs
    # return set(map(str, invs_symb))
    # return {"raw_invs": set(invs), "pred_invs": invs_symb_strs}
    return invs
