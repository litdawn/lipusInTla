import os

from PT_generators.RL_Prunning.PT_generator import PT_generator
# from PT_generators.simple_generator import PT_generator
from seedTemplate.tlaParser import tlaparser

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
    tla_ins, seed_tmpl = tlaparser.main_from_json(path2cfg, path2json, path2config)

    pt_generator = PT_generator(seed_tmpl, name)

    tmp_bench = os.path.join(os.getcwd(), "Benchmarks")
    checker = Checker(name, seed_tmpl.json_data, os.path.join(tmp_bench, "Result"))

    solved, ctis = checker.check_induction({})
    log.info(f"Begin_process:   {path2tla}")
    iteration = 0
    timer.new_timer(TIMER.EPOCH)

    while not solved:
        if iteration % 10 == 0:
            print(pt_generator.reward_list)

        current_time = timer.get_and_refesh(TIMER.EPOCH)
        log.info(f"第{iteration}轮已经结束，本轮耗时{current_time}，总耗时{timer.get_time(TIMER.TOTAL)}")
        if current_time >= config.Limited_time:
            log.error("Loop invariant Inference is OOT")
            return None, None
        iteration += 1

        log.info("==============================================================================================")
        log.info(f">>>iteration {iteration}: 开始寻找下一个引理不变式")

        timer.new_timer(TIMER.LEMMA_GENERATOR)
        candidate, lemmas = pt_generator.generate_next(ctis)  # candidate是候选归纳不变式， lemma是新生成的引理不变式候选者字典
        if len(lemmas) == 0:
            log.error("没找到下一个引理不变式")
            continue
        log.info(
            f">>>iteration {iteration}: 找到了一些候选者 {lemmas.keys()}，"
            f"花费{timer.get_time(TIMER.LEMMA_GENERATOR)}，开始不变式检查")
        timer.new_timer(TIMER.LEMMA_CHECKER)

        wrong_lemma = dict()
        is_inv, violate_dict = checker.check_invariants(lemmas)
        is_inv_names = list(is_inv.keys())
        # for inv_name, s in lemmas.items():
        #     if inv_name not in is_inv_names:
        #         wrong_lemma.update({inv_name: -2})  # todo 具体赋值

        if len(is_inv) < 1:
            log.info(f">>>iteration {iteration}: 均没通过不变式检查，花费{timer.get_time(TIMER.LEMMA_CHECKER)}")
            pt_generator.punish('VERY', violate_dict)
            continue
        log.info(
            f">>>iteration {iteration}: 不变式检测结束，{is_inv_names}是不变式，花费{timer.get_time(TIMER.LEMMA_CHECKER)}")

        timer.new_timer(TIMER.CTI_ELIMINATOR)
        add2can = checker.check_deduction(candidate, is_inv)
        add2can_name = list(add2can.keys())
        eliminate_num = dict()
        if len(add2can) != 0:
            add2can = prune(add2can)
            new_eliminate_ctis = checker.eliminate_ctis(candidate, add2can, ctis)
            log.info(f"消除cti阶段花费{timer.get_time(TIMER.CTI_ELIMINATOR)}")
            new_eliminate_ctis_name = list(new_eliminate_ctis.keys())

            for inv_name, e_cti in add2can.items():
                if inv_name in new_eliminate_ctis_name:
                    old_len = len(ctis)
                    ctis = del_from_ctis(ctis, new_eliminate_ctis[inv_name])
                    eliminate_num.update({inv_name: old_len - len(ctis)})
                eliminate_num.update({inv_name: 6})
        else:
            log.info(f"消除cti阶段花费{timer.get_time(TIMER.CTI_ELIMINATOR)}")

        for inv_name in is_inv_names:
            if inv_name not in add2can_name:
                wrong_lemma.update({inv_name: -2})  # todo 具体赋值

        if len(add2can) == 0 and len(ctis) > 0:
            log.info(f">>>iteration {iteration}: 都被之前的candidate蕴含了，应该严格一点")
            pt_generator.punish('LITTLE', wrong_lemma)
            continue
        elif len(add2can) > 0 and len(ctis) > 0:
            log.info(f">>>iteration {iteration}: 找到了{len(add2can)}个正确的不变式{add2can.keys()}，继续")
            pt_generator.prise("LITTLE", eliminate_num)
        elif len(ctis) == 0:
            log.info(f">>>iteration {iteration}: 看起来cti没了，检测归纳不变式&开始生成更多的cti")
        timer.new_timer(TIMER.CTI_GENERATOR)
        candidate.update(add2can)
        solved, new_ctis = checker.check_induction(candidate)
        if solved:
            log.info("找到结果")
            save_result(candidate)
            break
        ctis = ctis.union(new_ctis)
        log.info(
            f">>>iteration {iteration}: 找到了{len(new_ctis)}个CTI, 目前CTI的总数是{len(ctis)}，"
            f"花费{timer.get_time(TIMER.CTI_GENERATOR)}")
    # print(pt_generator.loss_list)
    # print(pt_generator.reward_list)
    # print(iteration)
    try:
        analyst = Analyst(accuracy_list=pt_generator.reward_list, loss_list=pt_generator.loss_list, iteration=iteration)
    finally:
        return


def del_from_ctis(orig_k_ctis, eliminated_ctis):
    new_ctis = set()
    for cti in orig_k_ctis:
        hashed = str(hash(cti))
        if hashed not in eliminated_ctis:
            new_ctis.add(cti)
    return new_ctis


def prune(inv_dict):
    prune_content = {}
    prune_result = {}

    def count_and(item):
        return item.count('\\/')

    def find_values_with_substring(target):
        for value_list in prune_content.values():
            for v in value_list:
                if target.find(v) == -1:
                    return False
        return True

    def deal_with_content(inv_content):
        inv_str = inv_content.split(":")[-1].strip()
        inv_list = inv_str.split("\\/")
        for index, inv in enumerate(inv_list):
            inv_list[index] = inv_list[index].strip()
        return inv_list

    sorted_dict = sorted(inv_dict.items(), key=lambda x: count_and(x[1]), reverse=False)
    min_num = 2
    for i, (inv_name, content) in enumerate(sorted_dict):
        if i == 0:
            min_num = count_and(content)
        if count_and(content) == min_num:
            prune_content.update({inv_name: deal_with_content(content)})
        elif find_values_with_substring(content):
            continue
        else:
            prune_content.update({inv_name: deal_with_content(content)})
    for inv_name in prune_content.keys():
        prune_result.update({inv_name: inv_dict[inv_name]})

    del inv_dict
    return prune_result


if __name__ == "__main__":
    begin = timer.new_timer("total")
    name = "simple_decentralized_lock"
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
