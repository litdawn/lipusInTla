# This is a Template Iteration PT generator which try from the simplest.
import random
import logging
# import pyeda

from seedTemplate.tlaParser.tlaparser import get_varnames_from_source_code
from itertools import combinations, combinations_with_replacement
from z3 import *


class PT_generator:
    s_id = 0

    def generate_invs(preds, num_invs, min_num_conjuncts=2, max_num_conjuncts=2,
                      process_local=False, boolean_style="tla", quant_vars=[]):
        """ Generate 'num_invs' random invariants with the specified number of conjuncts. """
        # Pick some random conjunct.
        # OR and negations should be functionally complete
        symb_neg_op = "~"
        if boolean_style == "cpp":
            # ops = ["&&", "||"]
            ops = ["||"]
            andop = "&&"
            neg_op = "!"
        elif boolean_style == "tla":
            # ops = ["/\\", "\\/"]
            ops = ["\\/"]
            andop = "/\\"
            neg_op = "~"

        # Assign a numeric id to each predicate.
        pred_id = {p: k for (k, p) in enumerate(preds)}

        invs = []
        invs_symb = []
        invs_symb_strs = []
        for invi in range(num_invs):
            conjuncts = list(preds)
            # conjuncts = list(map(str, range(len(preds))))
            num_conjuncts = random.randint(min_num_conjuncts, max_num_conjuncts)

            # Select first atomic term of overall predicate.
            c = random.choice(conjuncts)
            conjuncts.remove(c)

            # Optionally negate it.
            negate = random.choice([True, False])
            (n, fn) = (neg_op, symb_neg_op) if negate else ("", "")

            inv = n + "(" + c + ")"
            pred_id_var = f"x_{str(pred_id[c]).zfill(3)}"
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

                # Symbolic version of the predicate. Used for quickly
                # detecting logically equivalent predicate forms.
                pred_id_var = f"x_{str(pred_id[c]).zfill(3)}"
                symb_inv_str = fn + "(" + pred_id_var + ")" + " " + fop + " (" + symb_inv_str + ")"

            if inv not in invs:
                invs.append(inv)
                invs_symb.append(pyeda.inter.expr(symb_inv_str))
                # print(type(invs_symb[-1]))
                invs_symb_strs.append(symb_inv_str)
 
        logging.info(f"number of invs: {len(invs)}")

        # Do CNF based equivalence reduction.
        # invs = symb_equivalence_reduction(invs, invs_symb)
        logging.info(f"number of invs post CNF based equivalence reduction: {len(invs)}")

        # if len(quant_vars):
        # invs = pred_symmetry_reduction(invs, quant_vars)
        logging.info(f"number of post symmetry invs: {len(invs)}")
        return {"raw_invs": set(invs), "pred_invs": invs_symb_strs}
    def OnePass(self, var_names):
        res = []
        expers = []
        for varN in range(1, min(4, len(var_names) + 1)):
            for varlist in combinations(var_names, varN):
                # print(var_names, varN)
                exp = 0
                for varer in varlist:
                    exp += Int(varer) * Int('const_' + str(self.s_id))
                    self.s_id += 1
                res.append(simplify(exp) <= Int('const_' + str(self.s_id)))
                expers.append(simplify(exp))
                self.s_id += 1
        #Non-linear
        res_2 = []
        for exp_two in combinations_with_replacement(expers, 2):
            newExp = exp_two[0] * exp_two[1]
            res_2.append(simplify(newExp) <= Int('const_' + str(self.s_id)))
        res.extend(res_2)
        print(res[2])
        return res

    def generate_Exp_from_varnames(self, var_names):
        res = []
        res.extend(self.OnePass(var_names))
        return res

    def CNF(self, exps):
        exp_C = True
        for dnormExp in exps:
            exp_d = False
            for exp in dnormExp:
                exp_d = Or(exp_d, exp)
            try:
                exp_d = simplify(exp_d)
            except Exception as e:
                print(e)
            exp_C = And(exp_C, exp_d)
        try:
            exp_C = simplify(exp_C)
        except Exception as e:
            print(e)
        return exp_C

    def __init__(self, path2tla):
        self.var_names, self.tla = get_varnames_from_source_code(path2tla)
        self.exps = self.generate_Exp_from_varnames(self.var_names)
        self.used = set()
        self.lastPT = None

    def generate_next(self, CE):
        for n_c in range(1, 5):
            for n_d in range(1, 3):
                eeee = self.exps[:]
                eeee_i = 0
                if n_c * n_d > len(eeee):
                    print("Nothing to do about it")
                PT = None
                while PT is None or PT in self.used:
                    expser = []
                    if len(eeee) - eeee_i < n_c * n_d:
                        break
                    for x in range(n_c):
                        e_y = []
                        for y in range(n_d):
                            e_y.append(eeee[eeee_i])
                            eeee_i += 1
                        expser.append(e_y)
                    PT = self.CNF(expser)
                if PT in self.used or PT is None:
                    continue
                else:
                    self.lastPT = PT
                    return PT

    def punish(self, SorL, Deg, Whom):
        self.used.add(self.lastPT)

    def prise(self, Deg):
        pass

# 这是一个名为 PT_generator 的类，它试图从最简单的模板迭代生成 PT。这个类有两个方法：generate_invs 和 OnePass。
#
# generate_invs 方法生成具有指定数量联结词的随机不变式。它首先为每个谓词分配一个数字 ID，然后随机选择一些联结词，并可能对其进行否定。然后，它将这些联结词组合成一个不变式，并将其添加到不变式列表中。最后，它返回原始不变式和谓词不变式的集合。
#
# OnePass 方法接受一个变量名列表作为输入，然后对每个变量名生成一个表达式。这个表达式是该变量名与一个常数的乘积，这个常数是通过 self.s_id 生成的。每生成一个表达式，self.s_id 就会增加 1。这个方法的目的可能是为了生成一组基于输入变量的表达式，这些表达式可以用于进一步的计算或分析。
