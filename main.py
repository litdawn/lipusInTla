# A pipeline framework to realize the RL Pruning Tool for loop invariant inference
import os
import sys
import time
import logging

from PT_generators.RL_Prunning.PT_generator import PT_generator
from SMT_Solver.Config import config
from SMT_Solver.SMT_verifier import SMT_verifier
from seedTemplate.tlaParser import tlaparser
from SMT_Solver.cti_generator import generate_ctis


def main(path2tla, path2cfg, path2json):
    start_time = time.time()
    # Step 1. Input the three formation of the code.
    # todo: 第一步：tla静态检查

    # path2CFile, path2CFG, path2SMT = parseArgs()
    # Step 2. Load the Partial Template Generator.

    # todo: 第二步 生成seed和quants
    tla_ins, seed_tmpl = tlaparser.main(path2cfg, path2json)
    print(seed_tmpl.seeds)

    # # todo: 第三步
    #
    pT_generator = PT_generator(seed_tmpl)
    smt_verifier = SMT_verifier(tla_ins.variables)
    # Step 3. ENTER the ICE Solving Loop
    solved = False
    CE = {}
    # 原CE = {'p': [],'n': [],'i': []}
    logging.info("Begin_process:   ", path2tla)
    iteration = 0
    while not solved:
        current_time = time.time()
        if current_time - start_time >= config.Limited_time:
            logging.info("Loop invariant Inference is OOT")
            return None, None
        iteration += 1
        # Step 3.1 生成candidate
        candidate = pT_generator.generate_next(CE)
        if candidate is None:
            logging.info("The only way is to give up now")
            return None, None
        # Step 3.2 apalache验证
        # try:
        #     logging.info(f"find a candidate: {str(candidate)}")
        #     # raise TimeoutError # try this thing out
        # except TimeoutError as OOT:  # Out Of Time, we punish
        #     pT_generator.punish('STRICT', 'VERY', 'S')
        #     continue
        # if candidate is None:  # Specified too much, we loose.
        #     pT_generator.punish('LOOSE', 'MEDIUM', 'S')
        #     continue
        # # Step 3.3 Check if we bingo
        candidate = "\n/\\ ".join(candidate.values())
        logging.info(f"find a candidate: {str(candidate)}")

        try:
            is_right = smt_verifier.verify(candidate, path2tla)
        except TimeoutError as OOT:  # Out Of Time, we punish
            pT_generator.punish('STRICT', 'LITTLE', 'V')
            continue
        if is_right:  # Bingo, we prise
            solved = True
            logging.info("The answer is :  ", str(candidate))
            pT_generator.prise('VERY')
            current_time = time.time()
            logging.info("Time cost is :  ", str(current_time - start_time))
            return current_time - start_time, str(candidate)
        else:  # progressed anyway, we prise
            ctis, cti_time = generate_ctis(path2cfg, tla_ins)
            CE["i"].append(ctis)
            # if is_right.assignment not in CE[is_right.kind]:
            #     CE[is_right.kind].append(is_right.assignment)
            # pT_generator.prise('LITTLE')
            continue


if __name__ == "__main__":
    name = "learning_switch"
    path2tla = os.getcwd() + f"\\Benchmarks\\protocols\\{name}.tla"
    path2cfg = os.getcwd() + f"\\Benchmarks\\cfg\\{name}.cfg"
    path2json = os.getcwd() + f"\\Benchmarks\\json\\{name}.json"
    # path2config = path2tla[:-3] + "cfg"
    # command = f"java.exe -jar apalache-0.44.2/lib/apalache.jar check --inv=Inv --run-dir=gen_tla/apalache-cti-out --config={path2config} {path2tla} "
    # result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print(result)
    # path2CFG=r"Benchmarks/Linear/c_graph/2.c.json"
    # path2SMT=r"Benchmarks/Linear/c_smt2/2.c.smt"
    main(path2tla, path2cfg, path2json)
