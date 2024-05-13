# A pipeline framework to realize the RL Pruning Tool for loop invariant inference
import time
import logging
import os

# from PT_generators.RL_Prunning.PT_generator import PT_generator
from PT_generators.simple_generator import PT_generator
from seedTemplate.tlaParser import tlaparser

from SMT_Solver.cti_generator import generate_ctis
from SMT_Solver.cti_eliminator import eliminate_ctis
from SMT_Solver.inv_checker import check_invariants
from SMT_Solver.ind_checker import SMT_verifier
from SMT_Solver.Config import config

logging.basicConfig(level=logging.INFO)

def save_result(invs):
    str = "ind == \n "
    for inv_name, inv_content in invs.items():
        str += f"/\\ {inv_content} \n"
    with open(f"{os.getcwd()}/Benchmarks/Result/ind/{name}.txt", 'a') as f:
        f.write(str)
    pass
def main(path2tla, path2cfg, path2json, path2config):
    start_time = time.time()
    # Step 1. Input the three formation of the code

    # path2CFile, path2CFG, path2SMT = parseArgs()
    # Step 2. Load the Partial Template Generator.

    # tla_ins, seed_tmpl = tlaparser.main(path2cfg, path2json)
    tla_ins, seed_tmpl = tlaparser.main_from_json(path2cfg, path2json, path2config)

    pt_generator = PT_generator(seed_tmpl, name)
    smt_verifier = SMT_verifier(tla_ins.variables)
    # Step 3. ENTER the ICE Solving Loop
    solved = False
    ctis = set()
    # 原CE = {'p': [],'n': [],'i': []}
    logging.info(f"Begin_process:   {path2tla}")
    iteration = 0

    while not solved:
        current_time = time.time()
        if current_time - start_time >= config.Limited_time:
            logging.info("Loop invariant Inference is OOT")
            return None, None
        iteration += 1
        # Step 3.1 生成candidate
        candidate, lemmas, index = pt_generator.generate_next(ctis)
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
        logging.info("==============================================================================================")
        logging.info(f"step 1 iteration {iteration}: 找到了一些候选者 {str(candidate)}，开始检查")
        is_inv = list(check_invariants(lemmas, seed_tmpl=seed_tmpl))
        logging.info(f"find a {is_inv}")
        if len(is_inv) < 1:
            # 如果没通过不变式的检查，应该宽松一点
            logging.info(f">>>iteration {iteration}: 没通过不变式检查，宽松一点")
            pt_generator.punish('LOOSE', 'VERY')
            continue
        else:
            for lemma_name, lemma in candidate.items():
                if lemma_name != "Safety" and lemma_name != "Typeok" and ":" not in lemma:
                    candidate.update({lemma_name: seed_tmpl.quant_inv + lemma})
            # 如果被之前的candidate蕴含了，应该严格一点
            can2test = dict()
            for raw_name, raw_can in candidate.items():
                if raw_name in is_inv:
                    can2test.update({raw_name: raw_can})

            new_eliminate_cti, ctis = eliminate_ctis(candidate, can2test, ctis, seed_tmpl)
            logging.info(f"消除了一些cti，分别是{new_eliminate_cti}")
            # if (len(new_eliminate_cti) == 0 or len(new_eliminate_cti[is_inv[-1]]) == 0) and iteration > 1:

            if len(new_eliminate_cti) == 0 and len(ctis) > 0 and iteration > 1:
                logging.info(f">>>iteration {iteration}: 被之前的candidate蕴含了，应该严格一点")
                pt_generator.punish('STRICT', 'VERY')
                continue
            elif len(new_eliminate_cti) > 0 and len(ctis) > 0 and iteration > 1:
                logging.info(f">>>iteration {iteration}: 找到了一个正确的不变式，继续")
                pt_generator.prise('MEDIUM')
            elif len(ctis) == 0 and iteration > 1:
                candidate_str = "\n/\\ ".join(seed_tmpl.quant_inv + v for v in candidate.values())
                is_right = False
                pt_generator.prise('VERY')
                try:
                    logging.info(f">>>iteration {iteration}: {candidate_str}似乎是正确的归纳不变式，继续进行检查")
                    is_right = smt_verifier.verify(candidate_str, path2tla)
                except TimeoutError as OOT:  # Out Of Time, we punish
                    pt_generator.punish('STRICT', 'LITTLE')
                if is_right:  # Bingo, we prise
                    solved = True
                    logging.info("成功! The answer is :  ", candidate_str)
                    current_time = time.time()
                    logging.info("成功! Time cost is :  ", str(current_time - start_time))
                    return current_time - start_time, candidate_str
            logging.info(f">>>iteration {iteration}: 生成更多的cti")
            new_ctis = generate_ctis(seed_tmpl, candidate)
            old_len = len(ctis)
            ctis = ctis.union(new_ctis)
            if len(ctis) == old_len:
                logging.info("也许已经是结果了")
                save_result(candidate)
                break
            # ctis_str = "\n".join(cti.get_cti_state_string() for cti in ctis)
            logging.info(f">>>iteration {iteration}: 新找到了{len(new_ctis)}个CTI, 目前CTI的总数是{len(ctis)}")
            continue


if __name__ == "__main__":
    begin = time.time()
    name = "quorum_leader_election"
    config.specname = name
    path2tla = os.getcwd() + f"/Benchmarks/protocols/{name}.tla"
    path2cfg = os.getcwd() + f"/Benchmarks/cfg/{name}.cfg"
    path2json = os.getcwd() + f"/Benchmarks/json/{name}.json"
    path2config = os.getcwd() + f"/Benchmarks/{name}.config.json"
    config.TLC_PATH = os.path.join(os.getcwd(), "tla2tools.jar")
    # path2config = path2tla[:-3] + "cfg"
    # command = f"java.exe -jar apalache-0.44.2/lib/apalache.jar check --inv=Inv --run-dir=gen_tla/apalache-cti-out --config={path2config} {path2tla} "
    # result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print(result)
    # path2CFG=r"Benchmarks/Linear/c_graph/2.c.json"
    # path2SMT=r"Benchmarks/Linear/c_smt2/2.c.smt"
    main(path2tla, path2cfg, path2json, path2config)
    print(str(time.time()-begin))

