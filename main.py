import os
# import memory_profiler
from PT_generators.RL_Prunning.PT_generator import PT_generator
# from PT_generators.simple_generator import PT_generator
from seedTemplate.tlaParser import tlaparser
import torch
from SMT_Solver.Config import config

from checker.Checker import Checker

from Utilities.Timing import timer, TIMER
from Utilities.Logging import log
from Utilities.Analysis import Analyst


def save_result(invs):
    now_time = timer.get_time("total")
    strs = f"timecost: {now_time}s\nind == \n"
    for inv_name, inv_content in invs.items():
        strs += f"/\\ {inv_content} \n"
    with open(f"{os.getcwd()}/Benchmarks/Result/ind/{name}.txt", 'a') as f:
        f.write(strs)
    pass


# @memory_profiler.profile
def main(path2tla, path2cfg, path2json, path2config):
    tla_ins, seed_tmpl = tlaparser.main_from_json(path2cfg, path2json, path2config)

    pt_generator = PT_generator(seed_tmpl, name)

    tmp_bench = os.path.join(os.getcwd(), "Benchmarks")
    checker = Checker(name, seed_tmpl.json_data, os.path.join(tmp_bench, "Result"))

    lemmas_generate_num = []
    lemmas_is_invariant_num = []
    lemmas_add_to_ind_num = []
    # solved = False
    # ctis = checker.generate_cti({})
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
        lemmas_generate_num.append(len(lemmas))
        if len(lemmas) == 0:
            log.error("没找到下一个引理不变式")
            continue
        log.info(
            f">>>iteration {iteration}: 找到了一些候选者 {lemmas.keys()}，"
            f"花费{timer.get_time(TIMER.LEMMA_GENERATOR)}s，开始不变式检查")

        timer.new_timer(TIMER.LEMMA_CHECKER)
        is_inv, violate_dict = checker.check_invariants(lemmas)
        lemmas_is_invariant_num.append(len(is_inv))
        if len(is_inv) < 1:
            log.info(f">>>iteration {iteration}: 均没通过不变式检查，花费{timer.get_time(TIMER.LEMMA_CHECKER)}")
            lemmas_add_to_ind_num.append(0)
            pt_generator.punish('VERY', violate_dict)
            continue
        log.info(
            f">>>iteration {iteration}: 不变式检测结束，{list(is_inv.keys())}是不变式，花费{timer.get_time(TIMER.LEMMA_CHECKER)}")

        timer.new_timer(TIMER.CTI_ELIMINATOR)
        add2can = prune(is_inv)
        # print(is_inv)
        log.info(
            f">>>iteration {iteration}: 候选者精化，{len(add2can)}/{len(is_inv)}个不变式")
        # add2can = checker.check_deduction(candidate, add2can)

        eliminate_num = dict()
        new_add2can = {}
        if len(add2can) != 0:
            # new_eliminate_ctis = checker.eliminate_ctis(candidate, add2can, ctis)
            new_eliminate_ctis = checker.eliminate_ctis_without_chunk(candidate, add2can, ctis)
            log.info(f"消除cti阶段花费{timer.get_time(TIMER.CTI_ELIMINATOR)}s")
            new_eliminate_ctis_name = list(new_eliminate_ctis.keys())

            for inv_name, e_cti in add2can.items():
                if inv_name in new_eliminate_ctis_name:
                    old_len = len(ctis)
                    ctis = del_from_ctis(ctis, new_eliminate_ctis[inv_name])
                    per_eliminate_num = old_len - len(ctis)
                    if per_eliminate_num > 0:
                        eliminate_num.update({inv_name: old_len - len(ctis)})
                        new_add2can[inv_name] = add2can[inv_name]
            print(eliminate_num)
        else:
            log.info(f"消除cti阶段花费{timer.get_time(TIMER.CTI_ELIMINATOR)}s, 没有cti被消除")

        wrong_lemma = dict()
        for inv_name in is_inv:
            if inv_name not in new_add2can:
                wrong_lemma.update({inv_name: -1})

        lemmas_add_to_ind_num.append(len(new_add2can))
        # if len(ctis) < 5:
        #     for cti in ctis:
        #         print(cti)

        if len(new_add2can) == 0 and len(ctis) > 0:
            log.info(f">>>iteration {iteration}: 都被之前的candidate蕴含了，应该严格一点")
            pt_generator.punish('LITTLE', wrong_lemma)
            continue
        elif len(new_add2can) > 0 and len(ctis) > 0:
            log.info(
                f">>>iteration {iteration}: 找到了{len(new_add2can)}个正确的不变式{new_add2can.keys()}"
                f"，目前有{len(ctis)}个CTI，继续")
            pt_generator.prise("LITTLE", eliminate_num)
        elif len(ctis) == 0:
            log.info(f">>>iteration {iteration}: 看起来cti没了，检测归纳不变式&开始生成更多的cti")
            timer.new_timer(TIMER.CTI_GENERATOR)
            candidate.update(new_add2can)
            # new_ctis = checker.generate_cti(candidate)
            # if len(new_ctis) == 0:
            solved, new_ctis = checker.check_induction(candidate)
            if solved:
                log.info("找到结果")
                save_result(candidate)
                break
            ctis = ctis.union(new_ctis)
            log.info(
                f">>>iteration {iteration}: 找到了{len(new_ctis)}个CTI, 目前CTI的总数是{len(ctis)}，"
                f"花费{timer.get_time(TIMER.CTI_GENERATOR)}")

    print(lemmas_generate_num)
    print(lemmas_is_invariant_num)
    print(lemmas_add_to_ind_num)
    # print(pt_generator.loss_list)
    # print(pt_generator.reward_list)
    # print(iteration)
    # try:
    #     analyst = Analyst(accuracy_list=pt_generator.reward_list, loss_list=pt_generator.loss_list, iteration=iteration)
    # finally:
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
    MAX_LENGTH = 2
    MIN_LENGTH= 1

    def count_and(item):
        return item.count('\\/')

    def find_values_with_substring(target):
        if count_and(target) > MAX_LENGTH:
            return True
        have_find = True
        for value_list in prune_content.values():
            for v in value_list:
                if target.find(v) == -1:
                    have_find = False
                    break
            if have_find is True:
                return True
        return False

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
            min_num = max(count_and(content), MIN_LENGTH)
        if count_and(content) == min_num:
            prune_content.update({inv_name: deal_with_content(content)})
        elif count_and(content) < min_num or count_and(content)> MAX_LENGTH:
            continue
        elif count_and(content) > min_num and find_values_with_substring(content):
            continue
        else:
            prune_content.update({inv_name: deal_with_content(content)})
    for inv_name in prune_content.keys():
        prune_result.update({inv_name: inv_dict[inv_name]})

    return prune_result


if __name__ == "__main__":
    # test_prune()
    print(torch.cuda.is_available())
    begin = timer.new_timer("total")
    name = "two_phase_commit"
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

# def test_prune():
#     list_input = {
#         'Inv_0_3': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Commit"] \\in msgs)',
#         'Inv_0_12': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs)',
#         'Inv_0_13': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_14': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs)',
#         'Inv_0_21': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Commit"] \\in msgs)',
#         'Inv_0_27': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_28': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_29': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_30': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_33': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_36': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_37': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_38': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_39': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_40': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_41': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_42': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_43': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_44': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_45': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_46': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_47': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_48': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_51': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_57': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Commit"] \\in msgs)',
#         'Inv_0_66': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs)',
#         'Inv_0_67': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_68': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs)',
#         'Inv_0_75': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Commit"] \\in msgs)',
#         'Inv_0_81': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_82': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_83': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_85': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_88': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_90': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_91': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_92': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_93': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_94': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_95': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_96': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_97': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_98': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_99': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_100': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_101': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_103': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_106': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_108': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_109': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_110': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_112': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_115': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_117': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_118': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_119': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_120': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_121': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_122': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_123': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_124': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_125': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_126': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_127': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_128': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_130': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_133': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_135': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_136': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_137': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_139': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_142': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_144': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_145': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_146': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_147': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_148': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_149': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_150': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_151': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_152': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_153': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_154': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_155': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmj] = "committed")',
#         'Inv_0_157': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_160': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmj] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_174': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs)',
#         'Inv_0_175': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_176': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs)',
#         'Inv_0_189': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_190': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_191': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_198': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_199': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_200': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_201': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_202': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_203': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_204': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_205': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_206': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_207': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_208': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_209': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~(rmState[rmi] = "committed")',
#         'Inv_0_228': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs)',
#         'Inv_0_229': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_230': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "working") \\/ (rmState[rmj] = "aborted") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs)',
#         'Inv_0_246': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmi] = "committed") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_247': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmi] = "committed") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_248': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmi] = "committed") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_255': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_256': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_257': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_264': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_265': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_266': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmi] = "committed") \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_270': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_271': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_272': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_273': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_274': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_275': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_276': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_277': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_278': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_279': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_280': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_281': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_282': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_283': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_284': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_285': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_286': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_287': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_288': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_289': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_290': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Commit"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_291': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_292': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_293': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_294': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_295': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_296': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~(rmState[rmi] = "committed") \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_300': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_301': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_302': '\\A rmi \\in RM : \\A rmj \\in RM :  ([type |-> "Abort"] \\in msgs) \\/ (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_309': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_310': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working") \\/ ~(rmState[rmj] = "prepared")',
#         'Inv_0_311': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Abort"] \\in msgs) \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_318': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ (rmState[rmj] = "prepared") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working")',
#         'Inv_0_319': '\\A rmi \\in RM : \\A rmj \\in RM :  (rmState[rmj] = "aborted") \\/ (rmState[rmj] = "committed") \\/ ~([type |-> "Commit"] \\in msgs) \\/ ~(rmState[rmi] = "working") \\/ ~(rmState[rmj] = "prepared")'}
#     print(f"{len(list_input)}/{len(prune(list_input))}")
