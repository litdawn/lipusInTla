import json
import time
import logging
import subprocess
import tempfile
from SMT_Solver.Config import config
import sys
import os
from SMT_Solver.Util import *


def make_indquickcheck_tla_spec(specname, invs: dict, sat_invs_group: list, orig_k_ctis: set, seed_tmpl):
    # print(f"sat_invs_group {sat_invs_group}")
    begin = time.time()
    # invs_sorted = sorted(invs)

    # Start building the spec.
    # invcheck_tla_indcheck="---- MODULE %s_IndQuickCheck ----\n" % self.specname
    invcheck_tla_indcheck = "---- MODULE %s ----\n" % specname
    invcheck_tla_indcheck += "EXTENDS %s,Naturals,TLC\n\n" % config.specname

    invcheck_tla_indcheck += seed_tmpl.model_consts + "\n"

    # Create a variable to represent the value of each invariant.
    for inv in sat_invs_group:
        invi = int(inv.replace("inv_", "").strip())
        invname = "inv_%d" % invi
        invcheck_tla_indcheck += "VARIABLE %s_val\n" % invname
    invcheck_tla_indcheck += "VARIABLE ctiId\n"
    invcheck_tla_indcheck += "\n"

    # Add definitions for all invariants and strengthening conjuncts.
    for cinvname, cinvexp in invs.items():
        invcheck_tla_indcheck += ("%s == %s\n" % (cinvname, cinvexp))

    # for inv in sat_invs_group:
    #     invi = int(inv.replace("inv_", ""))
    #     invname = "inv_%d" % invi
    #     invexp = invs[invname]
    #     invcheck_tla_indcheck += ("%s == %s\n" % (invname, invexp))
    invcheck_tla_indcheck += "\n"

    kCTIprop = "kCTIs"
    invcheck_tla_indcheck += "%s == \n" % kCTIprop
    for cti in orig_k_ctis:
        # cti.getPrimedCTIStateString()
        invcheck_tla_indcheck += "\t\\/ " + cti.get_cti_state_string() + "\n"

        # Identify the CTI state by the hash of its string representation.
        invcheck_tla_indcheck += "\t   " + "/\\ ctiId = \"%d\"\n" % (hash(cti))

        # invcheck_tla_indcheck += "\n"
    invcheck_tla_indcheck += "\n"

    strengthening_conjuncts_str = ""
    for cinvname, cinvexp in invs.items():
        strengthening_conjuncts_str += "    /\\ %s\n" % cinvname

    invcheck_tla_indcheck += "\n"

    # TODO: Handle case of no satisfied invariants more cleanly.
    invcheck_tla_indcheck += "InvVals ==\n"
    invcheck_tla_indcheck += "\t    /\\ TRUE\n"
    for inv in sat_invs_group:
        invcheck_tla_indcheck += "\t   " + "/\\ %s_val = %s\n" % (inv, inv)
    invcheck_tla_indcheck += "\n"

    invcheck_tla_indcheck += "CTICheckInit ==\n"
    invcheck_tla_indcheck += "    /\\ %s\n" % kCTIprop
    invcheck_tla_indcheck += "    /\\ InvVals\n"
    invcheck_tla_indcheck += strengthening_conjuncts_str
    invcheck_tla_indcheck += "\n"

    # Add next-state relation that leaves the auxiliary variables unchanged.
    invcheck_tla_indcheck += "CTICheckNext ==\n"
    invcheck_tla_indcheck += "    /\\ NextUnchanged\n"
    invcheck_tla_indcheck += "    /\\ UNCHANGED ctiId\n"
    for inv in sat_invs_group:
        invcheck_tla_indcheck += "    /\\ UNCHANGED %s_val\n" % inv

    invcheck_tla_indcheck += "===="

    return invcheck_tla_indcheck


def make_ctiquickcheck_cfg(seed_tmpl):
    # Generate config file.
    invcheck_tla_indcheck_cfg = "INIT CTICheckInit\n"
    invcheck_tla_indcheck_cfg += "NEXT CTICheckNext\n"
    # invcheck_tla_indcheck_cfg += self.state_constraint
    invcheck_tla_indcheck_cfg += "\n"
    invcheck_tla_indcheck_cfg += seed_tmpl.constants
    invcheck_tla_indcheck_cfg += "\n"

    # for inv in sat_invs_group:
    #     invi = int(inv.replace("Inv",""))
    #     invname = "Inv%d" % invi
    #     invcheck_tla_indcheck_cfg += ("INVARIANT %s\n" % invname)

    return invcheck_tla_indcheck_cfg


def eliminate_ctis(invs, orig_k_ctis, seed_tmpl):
    # print(invs)
    # print(orig_k_ctis)
    tla_ins = seed_tmpl.tla_ins
    """ Check which of the given satisfied invariants eliminate CTIs. """

    # Save CTIs, indexed by their hashed value.
    cti_table = {}
    # print(orig_k_ctis)
    for cti in orig_k_ctis:
        hashed = str(hash(cti))
        cti_table[hashed] = cti

    eliminated_ctis = set()

    # Parameters for invariant generation.
    # min_conjs = 2
    # max_conjs = 2
    # process_local = False
    # quant_inv_fn = seed_tmpl.quant_inv

    iteration = 1
    # uniqid = 0
    # logging.info("\n>>> Iteration %d (num_conjs=(min=%d,max=%d),process_local=%s)" % (
    #     iteration, min_conjs, max_conjs, process_local))

    logging.info("Starting iteration %d of eliminate_ctis")

    print_invs = True  # disable printing for now.
    if print_invs:
        for inv, invexp in invs.items():
            invi = int(inv.replace("inv_", ""))
            invname = "inv_%d" % invi
            logging.info("%s %s %s", invname, "==", invexp)

    # Try to select invariants based on size ordering.
    # First, sort invariants by the number of CTIs they eliminate.

    # Cache all generated invariants so we don't need to unnecessarily re-generate them
    # in future rounds.
    # self.all_sat_invs = self.all_sat_invs.union(set(map(get_invexp, list(sat_invs))))
    # self.all_checked_invs = self.all_checked_invs.union(set(map(quant_inv_fn, list(invs))))
    # logging.info(f"Total number of unique satisfied invariants generated so far: {len(all_sat_invs)}")
    # logging.info(f"Total number of unique checked invariants so far: {len(all_checked_invs)}")

    #############
    #
    # For each invariant we generated, we want to see what set of CTIs it removes, so that we
    # can better decide which invariant to pick as a new strengthening conjunct. That's the goal
    # of the prototype code below.
    #
    ############

    logging.info("Checking which invariants eliminate CTIs.")

    # Initialize table mapping from invariants to a set of CTI states they violate.
    cti_states_eliminated_by_invs = {}
    for inv in invs.keys():
        cti_states_eliminated_by_invs[inv] = set()

    # Create metadir if necessary.
    os.system("mkdir -p states")

    #
    # Generate specs for checking CTI elimination with TLC. Note that we
    # partition the invariants into sub groups for checking with TLC, since
    # it can get overwhelmed when trying to check too many invariants at
    # once.
    #
    # TODO: Properly parallelize CTI elimination checking.
    #
    curr_ind = 0

    # Run CTI elimination checking in parallel.
    n_tlc_workers = 4
    # inv_chunks = list(chunks(sat_invs, n_tlc_workers))
    cti_chunks = list(chunks(list(orig_k_ctis), n_tlc_workers))
    inv_list = list(invs.keys())

    # while curr_ind < len(inv_list):
    # print(invs)
    # sat_invs_group = inv_list[curr_ind:(curr_ind + MAX_INVS_PER_GROUP)]
    logging.info("Checking invariant group of size %d (starting invariant=%d) for CTI elimination." % (
        len(inv_list), curr_ind))
    tlc_procs = []

    # Create the TLA+ specs and configs for checking each chunk.
    for ci, cti_chunk in enumerate(cti_chunks):
        # Build and save the TLA+ spec.
        spec_name = f"{config.specname}_chunk{ci}_IndQuickCheck"
        spec_str = make_indquickcheck_tla_spec(spec_name, invs, inv_list, cti_chunk, seed_tmpl)

        ctiquicktlafile = f"{os.path.join(config.specdir, config.GEN_TLA_DIR)}\\cti\\{spec_name}.tla"
        ctiquicktlafilename = f"{config.GEN_TLA_DIR}\\cti\\{spec_name}.tla"

        f = open(ctiquicktlafile, 'w')
        f.write(spec_str)
        f.close()

        # Generate config file.
        ctiquickcfgfile = f"{os.path.join(config.specdir, config.GEN_TLA_DIR)}\\cti\\{config.specname}_chunk{ci}_CTIQuickCheck.cfg"
        ctiquickcfgfilename = f"{config.GEN_TLA_DIR}\\cti\\{config.specname}_chunk{ci}_CTIQuickCheck.cfg"
        cfg_str = make_ctiquickcheck_cfg(seed_tmpl)

        f = open(ctiquickcfgfile, 'w')
        f.write(cfg_str)
        f.close()

        cti_states_file = os.path.join(config.specdir, f"states\\cti_quick_check_chunk{ci}_{curr_ind}.json")
        cti_states_relative_file = f"states\\cti_quick_check_chunk{ci}_{curr_ind}.json"

        # Run TLC.
        # Create new tempdir to avoid name clashes with multiple TLC instances running concurrently.
        dirpath = tempfile.mkdtemp()
        cmd = (f'{config.JAVA_EXE} -Xss16M -Djava.io.tmpdir="{dirpath}" -cp {config.TLC_PATH} tlc2.TLC '
               f'-maxSetSize {config.TLC_MAX_SET_SIZE} -dump json {cti_states_relative_file} '
               f'-noGenerateSpecTE -metadir states\\ctiquick_{config.specname}_chunk{ci}_{curr_ind} '
               f'-continue -checkAllInvariants -deadlock '
               f'-workers 1 -config {ctiquickcfgfilename} {ctiquicktlafilename}')

        logging.info("TLC command: " + cmd)
        workdir = None if config.specdir == "" else config.specdir
        subproc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=workdir)
        # time.sleep(0.25)
        tlc_procs.append(subproc)

    for ci, subproc in enumerate(tlc_procs):
        logging.info("Waiting for TLC termination " + str(ci))

        subproc.wait()
        ret = subproc.stdout.read().decode(sys.stdout.encoding)
        # print(ret)

        # TODO: Determine cause of flakiness when reading JSON states file.
        # time.sleep(0.5)

        # print ret
        lines = ret.splitlines()
        lines = grep_lines("State.*|/\\\\", lines)

        cti_states_file = os.path.join(config.specdir, f"states\\cti_quick_check_chunk{ci}_{curr_ind}.json")
        # cti_states_relative_file = f"states/cti_quick_check_chunk{ci}_{curr_ind}.json"

        # logging.info(f"Opening CTI states JSON file: '{cti_states_file}'")
        fcti = open(cti_states_file)
        text = fcti.read()
        cti_states = json.loads(text)["states"]
        fcti.close()
        # print("cti states:", len(cti_states))

        # Record the CTIs eliminated by each invariant.
        for cti_state in cti_states:
            sval = cti_state["val"]
            ctiHash = sval["ctiId"]
            # for inv in sat_invs_group:
            # for inv in inv_chunks[ci]:
            for inv in inv_list:
                if not sval[inv + "_val"]:
                    cti_states_eliminated_by_invs[inv].add(ctiHash)

        # for inv in cti_states_eliminated_by_invs:
        #     if len(cti_states_eliminated_by_invs[inv]):
        #         invi = int(inv.replace("inv_", ""))
        #         invexp = invs[inv]
        logging.info(f"{cti_states_eliminated_by_invs}")

        # The estimated syntactic/semantic "cost" (i.e complexity) of an invariant expression.

        # Key function for sorting invariants by the number of new CTIs they eliminate.

        # sorted_invs = sorted(sat_invs, reverse=True, key=inv_sort_key)
        chosen_invs = []
        cti_states_eliminated_in_iter = 0

        num_ctis_remaining = len(list(cti_table.keys())) - len(eliminated_ctis)
        num_orig_ctis = len(list(cti_table.keys()))
        # duration = time.time() - tstart
        logging.info("[ End eliminate (took {:.2f} secs.) ]".format(iteration, ))
        logging.info("%d original CTIs." % num_orig_ctis)
        logging.info("%d new CTIs eliminated in this iteration." % cti_states_eliminated_in_iter)
        logging.info("%d total CTIs eliminated." % len(eliminated_ctis))
        logging.info("%d still remaining." % num_ctis_remaining)

        # end_timing_ctielimcheck()

    return cti_states_eliminated_by_invs