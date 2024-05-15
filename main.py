import os

from PT_generators.RL_Prunning.PT_generator import PT_generator
# from PT_generators.simple_generator import PT_generator
from seedTemplate.tlaParser import tlaparser

from SMT_Solver.cti_generator import generate_ctis
from SMT_Solver.cti_eliminator import eliminate_ctis
from SMT_Solver.inv_checker import check_invariants
from SMT_Solver.ind_checker import SMT_verifier
from SMT_Solver.Config import config

from Utilities.Timing import timer, TIMER
from Utilities.Logging import log
from Utilities.Analysis import Analyst


def save_result(invs):
    str = "ind == \n "
    for inv_name, inv_content in invs.items():
        str += f"/\\ {inv_content} \n"
    with open(f"{os.getcwd()}/Benchmarks/Result/ind/{name}.txt", 'a') as f:
        f.write(str)
    pass


def main(path2tla, path2cfg, path2json, path2config):
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
    log.info(f"Begin_process:   {path2tla}")
    iteration = 0
    timer.new_timer(TIMER.EPOCH)
    reward_list = []
    while not solved:
        if iteration % 10 == 0:
            print(reward_list)
        current_time = timer.get_and_refesh(TIMER.EPOCH)
        log.info(f"第{iteration}轮已经结束，本轮耗时{current_time}")
        if current_time >= config.Limited_time:
            log.error("Loop invariant Inference is OOT")
            return None, None
        iteration += 1
        # Step 3.1 生成candidate
        log.info("==============================================================================================")
        log.info(f">>>iteration {iteration}: 开始寻找下一个引理不变式")
        timer.new_timer(TIMER.LEMMA_GENERATOR)
        candidate, lemmas, index = pt_generator.generate_next(ctis)
        if candidate is None:
            log.error("没找到下一个引理不变式")
            return None, None
        log.info(
            f">>>iteration {iteration}: 找到了一些候选者 {str(candidate)}，"
            f"花费{timer.get_time(TIMER.LEMMA_GENERATOR)}，开始不变式检查")
        timer.new_timer(TIMER.LEMMA_CHECKER)
        is_inv = list(check_invariants(lemmas, seed_tmpl=seed_tmpl))
        if len(is_inv) < 1:
            # 如果没通过不变式的检查，应该宽松一点
            log.info(f">>>iteration {iteration}: 没通过不变式检查，宽松一点，花费{timer.get_time(TIMER.LEMMA_CHECKER)}")
            reward_list.append(-10)
            pt_generator.punish('LOOSE', 'VERY')
            continue
        else:
            log.info(
                f">>>iteration {iteration}: 不变式检测结束，{is_inv}是不变式，花费{timer.get_time(TIMER.LEMMA_CHECKER)}")
            for lemma_name, lemma in candidate.items():
                if lemma_name != "Safety" and lemma_name != "Typeok" and ":" not in lemma:
                    candidate.update({lemma_name: seed_tmpl.quant_inv + lemma})
            # 如果被之前的candidate蕴含了，应该严格一点
            can2test = dict()
            for raw_name, raw_can in candidate.items():
                if raw_name in is_inv:
                    can2test.update({raw_name: raw_can})

            timer.new_timer(TIMER.CTI_ELIMINATOR)
            new_eliminate_cti, ctis = eliminate_ctis(candidate, can2test, ctis, seed_tmpl)
            log.info(f"消除了一些cti，分别是{new_eliminate_cti}，花费{timer.get_time(TIMER.CTI_ELIMINATOR)}")

            if len(new_eliminate_cti) == 0 and len(ctis) > 0 and iteration > 1:
                log.info(f">>>iteration {iteration}: 被之前的candidate蕴含了，应该严格一点")
                reward_list.append(-1)
                pt_generator.punish('STRICT', 'LITTLE')
                continue
            elif len(new_eliminate_cti) > 0 and len(ctis) > 0 and iteration > 1:
                log.info(f">>>iteration {iteration}: 找到了一个正确的不变式，继续")
                reward_list.append(5)
                pt_generator.prise('VERY')
            elif len(ctis) == 0 and iteration > 1:
                candidate_str = "\n/\\ ".join(seed_tmpl.quant_inv + v for v in candidate.values())
                is_right = False
                # reward_list.append(10)
                # pt_generator.prise('VERY')
                try:
                    log.info(f">>>iteration {iteration}: {candidate_str}似乎是正确的归纳不变式，继续进行检查")
                    is_right = smt_verifier.verify(candidate_str, path2tla)
                except TimeoutError as OOT:  # Out Of Time, we punish
                    pt_generator.punish('STRICT', 'LITTLE')
                if is_right:  # Bingo, we prise
                    save_result(candidate)
                    break

            log.info(f">>>iteration {iteration}: 开始生成更多的cti")
            timer.new_timer(TIMER.CTI_GENERATOR)
            new_ctis = generate_ctis(seed_tmpl, candidate)
            old_len = len(ctis)
            ctis = ctis.union(new_ctis)
            if len(ctis) == old_len:
                log.info("找到结果")
                save_result(candidate)
                break
            log.info(
                f">>>iteration {iteration}: 新找到了{len(new_ctis)}个CTI, 目前CTI的总数是{len(ctis)}，"
                f"花费{timer.get_time(TIMER.CTI_GENERATOR)}")
            continue
    # print(pt_generator.loss_list)
    print(reward_list)
    print(iteration)
    try:
        analyst = Analyst(accuracy_list=reward_list, loss_list=pt_generator.loss_list, iteration=iteration)
    finally:
        return



if __name__ == "__main__":
    begin = timer.new_timer("total")
    name = "lockserv"
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
    timer.new_timer(TIMER.TOTAL)
    main(path2tla, path2cfg, path2json, path2config)
    print(f"总耗时{timer.get_time(TIMER.TOTAL)}")
