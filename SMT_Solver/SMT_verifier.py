import logging

from z3 import *

set_param('parallel.enable', True)
from SMT_Solver.Config import config
from Utilities.TimeController import time_limit_calling
import subprocess

apalache_bin = "apalache-0.44.2/bin/apalache-mc"
# jvm_args = "JVM_ARGS='-Xss16M'"

cmd = 'java.exe -jar -Xss16M -Djava.io.tmpdir="test" -cp tla2tools-checkall.jar tlc2.TLC -continue -deadlock -config {path2config} {path2tla}'


class Counterexample:
    kind = "?"
    assignment = {}


class SMT_verifier:
    tpl = []

    # 读入模板 1.const 2. func 3. 三个安全条件
    # def initTpl(self, path2SMT):
    #     vc_sections = [""]
    #     with open(path2SMT, 'r') as vc:
    #         for vc_line in vc.readlines():
    #             if "SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop" in vc_line:
    #                 vc_sections.append("")
    #             else:
    #                 vc_sections[-1] += vc_line
    #     assert len(vc_sections) == 5
    #
    #     self.tpl = [vc_sections[0]]
    #
    #     for i in range(2, 5):
    #         self.tpl.append(vc_sections[1] + vc_sections[i])

    # def check_predicates(self, preds, tlc_workers=6)
    # 这里要换成tlaps

    def __init__(self, varnames):
        self.tla = []
        self.varnames = varnames
        self.replacement = "abcdefghijklmnopqrstuvwxyz"

    def verify(self, Can_I, path2tla):
        # 执行命令
        self.write2tla(Can_I, path2tla)
        path2config = path2tla[:-3] + "cfg"
        path2dump = path2tla[:path2tla.find("/")+1]+path2tla[path2tla.find("/")+1:-3]+"json"
        command = (f'java -cp tla2tools.jar tlc2.TLC -continue -deadlock -workers 4 -config {path2config} -dump {path2dump} {path2tla}')
        # command = f"java.exe -jar apalache-0.44.2/lib/apalache.jar check --inv=IndCand --run-dir=gen_tla/apalache-cti-out --config={path2config} {path2tla} "
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 输出结果
        print(result.stderr.decode("gbk"))
        print(result.stdout.decode("gbk"))
        if result.returncode == 0:
            return None
        else:
            counterexample = Counterexample()
            # counterexample.kind = kind
            # counterexample.assignment = ce
            return counterexample

    def write2tla(self, Can_I, path2tla):
        self.readTLA(path2tla)
        reg = "/const_[0-9]+/"
        self.tla[-2] = ("IndCand == " + str(Can_I) + "\n")
        with open(path2tla, 'w') as f:
            f.writelines(self.tla)

        return

    def smt2tla(self, can_I):
        can_I = str(can_I)
        begin = 0
        end = 0
        pre = ""
        for var in self.varnames:
            begin = end
            end += can_I.count(var)
            if end > begin:
                pre += "\A "
                for i in range(begin, end):
                    loc = can_I.find(var)
                    can_I = can_I[0:loc] + self.replacement[i] + can_I[loc + len(var):]
                    pre += self.replacement[i] + ","
                pre = pre[0:-1]
                pre += " \in " + var
        can_I = pre + ": " + can_I
        print(can_I)
        return can_I

    def readTLA(self, path2tla):
        tla_lines = []
        with open(path2tla, 'r') as f:
            for line in f.readlines():
                if line.startswith("===="):
                    tla_lines.append("\n")
                    tla_lines.append("====")
                elif line.startswith("EXTENDS") and line.count("Integers") == 0:
                    tla_lines.append(line[:-1] + ", Integers\n")
                elif not line.startswith("IndCand"):
                    tla_lines.append(line)

        self.tla = tla_lines
        return