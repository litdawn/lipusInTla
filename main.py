import os

from PT_generators.RL_Prunning.PT_generator import PT_generator
# from PT_generators.simple_generator import PT_generator
from seedTemplate.tlaParser import tlaparser

# from SMT_Solver.cti_generator import generate_ctis
# from SMT_Solver.cti_eliminator import eliminate_ctis
# from SMT_Solver.inv_checker import check_invariants
# from SMT_Solver.ind_checker import SMT_verifier
from SMT_Solver.Config import config

from checker.Checker import Checker

from Utilities.Timing import timer, TIMER
from Utilities.Logging import log
from Utilities.Analysis import Analyst


def save_result(invs):
    strs = "ind == \n "
    for inv_name, inv_content in invs.items():
        strs += f"/\\ {inv_content} \n"
    with open(f"{os.getcwd()}/Benchmarks/Result/ind/{name}.txt", 'a') as f:
        f.write(strs)
    pass


def main(path2tla, path2cfg, path2json, path2config):
    # Step 1. Input the three formation of the code

    # path2CFile, path2CFG, path2SMT = parseArgs()
    # Step 2. Load the Partial Template Generator.

    # tla_ins, seed_tmpl = tlaparser.main(path2cfg, path2json)
    tla_ins, seed_tmpl = tlaparser.main_from_json(path2cfg, path2json, path2config)

    pt_generator = PT_generator(seed_tmpl, name)
    tmp_bench = os.path.join(os.getcwd(), "Benchmarks")
    checker = Checker(name, seed_tmpl, os.path.join(tmp_bench, "Result"))
    # smt_verifier = SMT_verifier(tla_ins.variables)
    # Step 3. ENTER the ICE Solving Loop
    solved, ctis = checker.check_induction()
    # 原CE = {'p': [],'n': [],'i': []}
    log.info(f"Begin_process:   {path2tla}")
    iteration = 0
    timer.new_timer(TIMER.EPOCH)

    while not solved:
        if iteration % 10 == 0:
            print(pt_generator.reward_list)
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
        candidate, lemmas = pt_generator.generate_next(ctis)  # candidate是候选归纳不变式， lemma是新生成的引理不变式候选者字典
        if candidate is None:
            log.error("没找到下一个引理不变式")
            return None, None
        log.info(
            f">>>iteration {iteration}: 找到了一些候选者 {str(candidate)}，"
            f"花费{timer.get_time(TIMER.LEMMA_GENERATOR)}，开始不变式检查")
        timer.new_timer(TIMER.LEMMA_CHECKER)
        # is_inv = list(check_invariants(lemmas, seed_tmpl=seed_tmpl))
        is_inv = checker.check_invariants(lemmas)
        if len(is_inv) < 1:
            # 如果没通过不变式的检查，应该宽松一点
            log.info(f">>>iteration {iteration}: 均通过不变式检查，花费{timer.get_time(TIMER.LEMMA_CHECKER)}")
            pt_generator.punish('VERY', lemmas)
            continue
        else:
            log.info(
                f">>>iteration {iteration}: 不变式检测结束，{is_inv}是不变式，花费{timer.get_time(TIMER.LEMMA_CHECKER)}")
            # 找出不变式全文
            can2test = dict()
            for raw_name, raw_can in lemmas.items():
                if raw_name in is_inv:
                    can2test.update({raw_name: raw_can})

            timer.new_timer(TIMER.CTI_ELIMINATOR)
            new_eliminate_ctis, e_ctis = checker.check_deduction(candidate, can2test, ctis)
            log.info(f"消除了一些cti，分别是{new_eliminate_ctis}，花费{timer.get_time(TIMER.CTI_ELIMINATOR)}")

            delete_lemmas = dict()
            for inv_name, e_cti in e_ctis.items():
                if len(e_cti) == 0:
                    can2test.pop(inv_name)
                    delete_lemmas.update({inv_name: lemmas[inv_name]})

            if len(can2test) == 0 and len(ctis) > 0 and iteration > 1:
                log.info(f">>>iteration {iteration}: {delete_lemmas} 被之前的candidate蕴含了，应该严格一点")
                pt_generator.punish('LITTLE', delete_lemmas)
                continue
            elif len(can2test) > 0 and len(ctis) > 0 and iteration > 1:
                log.info(f">>>iteration {iteration}: 找到了{len(can2test)}个正确的不变式，继续")
                pt_generator.prise("LITTLE", can2test)

            log.info(f">>>iteration {iteration}: 开始生成更多的cti")
            timer.new_timer(TIMER.CTI_GENERATOR)
            solved, new_ctis = checker.check_induction(candidate.update(can2test))
            if solved:
                log.info("找到结果")
                save_result(candidate)
                break
            ctis = ctis.union(new_ctis)
            log.info(
                f">>>iteration {iteration}: 找到了{len(new_ctis)}个CTI, 目前CTI的总数是{len(ctis)}，"
                f"花费{timer.get_time(TIMER.CTI_GENERATOR)}")
    # print(pt_generator.loss_list)
    print(pt_generator.reward_list)
    print(iteration)
    try:
        analyst = Analyst(accuracy_list=reward_list, loss_list=pt_generator.loss_list, iteration=iteration)
    finally:
        return


if __name__ == "__main__":
    begin = timer.new_timer("total")
    name = ""
    benchmark_path = os.path.join(os.getcwd(), "Benchmarks")
    config.specname = name
    config.TLC_PATH = os.path.join(os.getcwd(), "tla2tools.jar")

    path2tla = os.path.join(os.path.join(benchmark_path, "protocols"), f"{name}.tla")
    path2cfg = os.path.join(os.path.join(benchmark_path, "cfg"), f"{name}.cfg")
    path2json = os.path.join(os.path.join(benchmark_path, "json"), f"{name}.json")
    path2config = os.path.join(benchmark_path, f"{name}.config.json")

    timer.new_timer(TIMER.TOTAL)
    main(path2tla, path2cfg, path2json, path2config)
    print(f"总耗时{timer.get_time(TIMER.TOTAL)}")
