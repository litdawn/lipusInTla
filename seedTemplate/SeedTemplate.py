import random
from itertools import combinations
from itertools import permutations

from seedTemplate.tlaParser.type import Type


class SeedTemplate:
    def __init__(self, tla_ins):
        self.quants = []
        self.seeds = []
        self.set = []
        self.array = []
        self.bool = []
        self.str = ["VARA", "VARB", "VARC", "VARD", "VARE"]
        self.def_var = ["VARA", "VARB", "VARC", "VARD", "VARE"]
        self.tla_ins = tla_ins

    def fill_str(self, tla_ins):
        cons = tla_ins.constants
        for con in iter(cons):
            self.set.append(con)
            # try:
            #     if con["self_type"] == Type.STRING:
            #         self.add2str_and_bool(con)
            #     pass
            # except Exception as e:
            #     if con["info"]["self_type"] == Type.SET:
            #         for sub_con in con["info"]["sub"]:
            #             self.add2str_and_bool({"name": "default", "self_type": con["info"]["sub_type"]})
            #     pass

    def add2str_and_bool(self, var):
        if var.self_type == Type.STRING:
            self.str.append(var.name)
        elif var.self_type == Type.BOOL:
            self.bool.append(var.name)

    # 主函数
    def generate(self):
        # 处理常量
        self.fill_str(self.tla_ins)
        # 处理变量
        for var in self.tla_ins.variables.values():
            self.generate_special_body(var)
        # 处理状态转换行为
        for var in self.tla_ins.actions.values():
            self.generate_special_body(var)
        # 生成"="
        self.generate_equal()
        # 生成\subseteq
        self.generate_subseteq()
        # 生成量词
        self.generate_quants()

        print("seeds", self.seeds)
        print("quants", self.quants)
        print("str", self.str)
        return self.seeds, self.quants

    def generate_special_body(self, var):
        if var.self_type == Type.SET:
            # do in
            self.set.append(var.name)
            self.generate_IN(var)
        elif var.self_type == Type.ARRAY:
            # do []
            self.generate_BOX(var)
        elif var.self_type == Type.ACTION:
            # do ()
            self.generate_action(var)
        elif var.self_type == Type.STRING or var.self_type == Type.BOOL:
            self.seeds.append(var.name)
            self.add2str_and_bool(var)
        return 0

    #  in
    def generate_IN(self, var):
        for i in combinations(self.def_var, var.sub_num):
            name = "<<" + ', '.join(i) + ">> \\in " + var.name
            self.seeds.append(name)
        return 0

    # []
    def generate_BOX(self, var):
        for i in self.str:
            if "[" not in i:
                name = var.name + "[" + i + "]"
                new_var = self.tla_ins.duplicate_var(name, var.content)
                self.generate_special_body(new_var)
        return 0

    # ()
    def generate_action(self, var):
        for i in combinations(self.str, var.param_num):
            name = var.name + "(" + ','.join(i) + ")"
            new_var = self.tla_ins.duplicate_var(name, var.result)
            self.generate_special_body(new_var)
        return 0

    def generate_equal(self):
        for i in combinations(self.str, 2):
            self.seeds.append("=".join(i))
        for i in combinations(self.set, 2):
            self.seeds.append("=".join(ele for ele in i))
        for i in self.set:
            # self.seeds.append(i.name + "=" + i.name)
            self.seeds.append(i + "= {}")
        return 0

    def generate_notequal(self):
        for i in self.str:
            for integer in range(0, 5):
                self.seeds.append(f"{i} >= {integer}")
                self.seeds.append(f"{i} <= {integer}")

    def generate_subseteq(self):
        for i in permutations(self.set, 2):
            self.seeds.append(" \\subseteq ".join(ele for ele in i))

    def generate_quants(self):
        for quant in self.def_var:
            for con in self.tla_ins.constants:
                # if con.self_type == Type.SET:
                self.quants.append(f"\\A {quant} in  {con} : ")
        return 0

    # 1. 生成seed
    #               ①\in
    #               ② arr的[]
    #               ③ =
    #               ④ \subseteq）
    #               ⑤state
    # 2. seed组合 （\and \or）
