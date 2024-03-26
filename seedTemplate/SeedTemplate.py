import random
from itertools import permutations
from tlaParser.type import Type


class SeedTemplate(object):
    # __OpDefNode = "@"
    # # Prefix operators
    # __OP_lnot = "\\lnot"
    # __OP_subset = "SUBSET"
    # __OP_union = "UNION"
    # __OP_domain = "DOMAIN"
    # __OP_box = "[]"
    # __OP_diamond = "<>"
    # __OP_enabled = "ENABLED"
    # __OP_unchanged = "UNCHANGED"
    # # Infix operators
    # __OP_eq = "="
    # __OP_land = "\\land"
    # __OP_lor = "\\lor"
    # __OP_implies = "=>"
    # __OP_cdot = "\\cdot"
    # __OP_equiv = "\\equiv"
    # __OP_leadto = "~>"
    # __OP_arrow = "-+->"
    # __OP_noteq = "/="
    # __OP_subseteq = "\\subseteq"
    # __OP_in = "\\in"
    # __OP_notin = "\\notin"
    # __OP_setdiff = "\\"
    # __OP_cap = "\\intersect"
    # __OP_cup = "\\union"
    # # Below are not built - in operators, but are useful definitions.
    # __OP_dotdot = ".."
    # __OP_plus = "+"
    # __OP_minus = "-"
    # __OP_times = "*"
    # __OP_lt = "<"
    # __OP_leq = "\\leq"
    # __OP_gt = ">"
    # __OP_geq = "\\geq"
    # # Postfix operators
    # __OP_prime = "'"
    #
    # # prefix_operators = [__OP_lnot, __OP_subset, __OP_union, __OP_domain, __OP_box, __OP_diamond, __OP_enabled,
    # #                     __OP_unchanged]
    # # infix_operators = [__OP_eq, __OP_land, __OP_lor, __OP_implies, __OP_cdot, __OP_equiv, __OP_leadto, __OP_arrow,
    # #                    __OP_noteq, __OP_subseteq, __OP_in, __OP_notin, __OP_setdiff, __OP_cap, __OP_cup]
    # # postfix_operators = [__OP_prime]
    # # math_operators = [__OP_dotdot, __OP_plus, __OP_minus, __OP_times, __OP_lt, __OP_leq, __OP_gt, __OP_geq]
    #
    # # var {} var
    # operators_1 = [__OP_eq, __OP_implies]
    # # var {} set
    # operators_2 = [__OP_in, __OP_notin, ]
    # # set {} set
    # operators_3 = [__OP_subseteq]
    # # seed {} seed
    # operators_4 = [__OP_land, __OP_lor]
    #
    # # # Opcodes of level 0 */
    # # OPCODE_bc = 1
    # # OPCODE_be = 2
    # # OPCODE_bf = 3
    # # OPCODE_case = 4
    # # OPCODE_cp = 5
    # # OPCODE_cl = 6
    # # OPCODE_dl = 7
    # # OPCODE_exc = 8
    # # OPCODE_fa = 9
    # # OPCODE_fc = 10
    # # OPCODE_ite = 11
    # # OPCODE_nrfs = 12
    # # OPCODE_pair = 13
    # # OPCODE_rc = 14
    # # OPCODE_rs = 15
    # # OPCODE_rfs = 16
    # # OPCODE_seq = 17
    # # OPCODE_se = 18
    # # OPCODE_soa = 19
    # # OPCODE_sor = 20
    # # OPCODE_sof = 21
    # # OPCODE_sso = 22
    # # OPCODE_tup = 23
    # # OPCODE_uc = 24
    # # OPCODE_ue = 25
    # # OPCODE_uf = 26
    # #
    # # # Prefix opcode of level 0
    # # OPCODE_lnot = OPCODE_uf + 1
    # # OPCODE_neg = OPCODE_uf + 2
    # # OPCODE_subset = OPCODE_uf + 3
    # # OPCODE_union = OPCODE_uf + 4
    # # OPCODE_domain = OPCODE_uf + 5
    # # OPCODE_enabled = OPCODE_uf + 8
    # #
    # # # Infix opcode of level 0
    # # OPCODE_eq = OPCODE_enabled + 1
    # # OPCODE_land = OPCODE_enabled + 2
    # # OPCODE_lor = OPCODE_enabled + 3
    # # OPCODE_implies = OPCODE_enabled + 4
    # # OPCODE_equiv = OPCODE_enabled + 5
    # # OPCODE_noteq = OPCODE_enabled + 6
    # # OPCODE_subseteq = OPCODE_enabled + 7
    # # OPCODE_in = OPCODE_enabled + 8
    # # OPCODE_notin = OPCODE_enabled + 9
    # # OPCODE_setdiff = OPCODE_enabled + 10
    # # OPCODE_cap = OPCODE_enabled + 11
    # # OPCODE_nop = OPCODE_enabled + 12
    # # OPCODE_cup = OPCODE_enabled + 13
    # #
    # # # Opcodes of level 2
    # # OPCODE_prime = OPCODE_cup + 1
    # # OPCODE_unchanged = OPCODE_cup + 2
    # # OPCODE_aa = OPCODE_cup + 3
    # # OPCODE_sa = OPCODE_cup + 4
    # # OPCODE_cdot = OPCODE_cup + 5
    # #
    # # # Opcodes of level 3
    # # OPCODE_sf = OPCODE_cdot + 1
    # # OPCODE_wf = OPCODE_cdot + 2
    # # OPCODE_te = OPCODE_cdot + 3
    # # OPCODE_tf = OPCODE_cdot + 4
    # # OPCODE_leadto = OPCODE_cdot + 5
    # # OPCODE_arrow = OPCODE_cdot + 6
    # # OPCODE_box = OPCODE_cdot + 7
    # # OPCODE_diamond = OPCODE_cdot + 8

    def __init__(self):
        self.quants = []
        self.seeds = []
        self.set = []
        self.array = []
        self.bool = []
        self.str = ["VARA", "VARB", "VARC", "VARD", "VARE"]
        self.def_var = ["VARA", "VARB", "VARC", "VARD", "VARE"]
        self.tla_ins = None

    def fill_str(self, tla_ins):
        cons = tla_ins.constants
        for con in cons:
            if con.self_type == Type.STRING:
                self.add2str_and_bool(con)
            elif con.self_type == Type.SET:
                for sub_con in con.sub:
                    self.add2str_and_bool(sub_con)

    def add2str_and_bool(self, var):
        if var.self_type == Type.STRING:
            self.str.append(var.name)
        elif var.self_type == Type.BOOL:
            self.bool.append(var.name)

    # 主函数
    def generate(self, tla_ins):
        self.tla_ins = tla_ins
        # 处理常量
        self.fill_str(tla_ins)
        # 处理变量
        for var in tla_ins.variables:
            self.generate_special_body(var)
        # 处理状态转换行为
        for action in tla_ins.actions:
            self.generate_special_body(action)
        # 生成"="
        self.generate_equal()
        # 生成量词
        self.generate_quants()

    def generate_special_body(self, var):
        if var.self_type == Type.SET:
            # do in
            self.set.append(var)
            self.generate_IN(var)
        elif var.self_type == Type.ARRAY:
            # do []
            self.generate_BOX(var)
        elif var.self_type == Type.ACTION:
            # do ()
            self.generate_action(var)
        elif var.self_type == Type.STRING:
            self.seeds.append(var.name)
            self.add2str_and_bool(var)
        elif var.self_type == Type.BOOL:
            self.seeds.append(var.name)
            self.add2str_and_bool(var.name)
        return 0

    #  in
    def generate_IN(self, var):
        for i in permutations(self.def_var, var.sub_num):
            name = "<<" + ', '.join(i) + ">> \\in " + var.name
            self.seeds.append(name)
        return 0

    # []
    def generate_BOX(self, var):
        for i in self.str:
            name = var.name + "[" + i + "]"
            new_var = self.tla_ins.construct_var(name, var.content)
            self.generate_special_body(new_var)
        return 0

    # ()
    def generate_action(self, var):
        for i in permutations(self.str, var.param_num):
            name = var.name + "(" + ','.join(i) + ")"
            new_var = self.tla_ins.construct_var(name, var.result)
            self.generate_special_body(new_var)
        return 0

    def generate_equal(self):
        for i in permutations(self.str, 2):
            self.seeds.append("=".join(i))
        return 0

    def generate_quants(self):
        for quant in self.def_var:
            for con in self.tla_ins.constants:
                if con.self_type == Type.SET:
                    self.quants.append("\\A " + quant + " in " + con.name)

        return 0

    # 1. 生成seed
    #               ①\in
    #               ② arr的[]
    #               ③ =
    #               ④ \subseteq）
    #               ⑤state
    # 2. seed组合 （\and \or）
