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
from SMT_Solver.inv_checker import check_invariants


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
    pt_generator = PT_generator(seed_tmpl, name)
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
        candidate, lemmas, index = pt_generator.generate_next(CE)
        if candidate is None:
            logging.info("The only way is to give up now")
            return None, None
        # Step 3.2 apalache验证
        # try:
        #     logging.info(f"find a candidate: {str(candidate)}")
        #     # raise TimeoutError # try this thing out
        # except TimeoutError as OOT:  # Out Of Time, we punish #超时，认为太宽松
        #     pt_generator.punish('STRICT', 'VERY', 'S')
        #     continue
        # if candidate is None:  # Specified too much, we loose. 没找到candidate，认为模板太严格
        #     pt_generator.punish('LOOSE', 'MEDIUM', 'S')
        #     continue
        # # Step 3.3 Check if we bingo

        logging.info(f"find a candidate: {str(candidate)}")
        print(str(candidate))

        is_inv = check_invariants(lemmas, specname=name, tla_ins=tla_ins)
        if len(is_inv) <= 2:
            # 如果没通过不变式的检查，应该宽松一点
            pt_generator.punish('LOOSE', 'VERY', 'V')
            continue

        try:
            candidate = "\n/\\ ".join(candidate.values())
            is_right = smt_verifier.verify(candidate, path2tla)
        except TimeoutError as OOT:  # Out Of Time, we punish
            pt_generator.punish('STRICT', 'LITTLE', 'V')
            continue
        if is_right:  # Bingo, we prise
            solved = True
            logging.info("The answer is :  ", str(candidate))
            pt_generator.prise('VERY')
            current_time = time.time()
            logging.info("Time cost is :  ", str(current_time - start_time))
            return current_time - start_time, str(candidate)
        else:
            # 如果被之前的candidate蕴含了，应该严格一点
            pt_generator.punish('STRICT', 'VERY', 'V')
            ctis, cti_time = generate_ctis(path2cfg, tla_ins)
            CE["i"].append(ctis)
            # if is_right.assignment not in CE[is_right.kind]:
            #     CE[is_right.kind].append(is_right.assignment)
            # pt_generator.prise('LITTLE')
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
