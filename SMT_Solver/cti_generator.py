import random
import os
import logging
import json
import sys
import subprocess
import time
import tempfile
import re
from SMT_Solver.Config import config


class CTI:
    """ Represents a single counterexample to induction (CTI) state. """

    def __init__(self, cti_str, cti_lines, action_name):
        self.cti_str = cti_str
        self.action_name = action_name
        self.cti_lines = cti_lines

    def getCTIStateString(self):
        return self.cti_str

    def getPrimedCTIStateString(self):
        """ Return CTI as TLA+ state string where variables are primed."""
        primed_state_vars = []
        for cti_line in self.cti_lines:
            # Remove the conjunction (/\) at the beginning of the line.
            cti_line = cti_line[2:].strip()
            # print(cti_line)
            # Look for the first equals sign.
            first_equals = cti_line.index("=")
            varname = cti_line[:first_equals].strip()
            varval = cti_line[first_equals + 1:]
            # print("varname:", varname)
            # print("varval:", varval)
            primed_state_vars.append(f"/\\ {varname}' ={varval}")
        primed_state = " ".join(primed_state_vars)
        # print(primed_state)
        return primed_state

    def getActionName(self):
        return self.action_name

    def setActionName(self, action_name):
        self.action_name = action_name

    def __hash__(self):
        return hash(self.cti_str)

    def __eq__(self, other):
        return hash(self.cti_str) == hash(other.cti_str)

    def __str__(self):
        return self.cti_str

    # Order CTIs as strings.
    def __lt__(self, other):
        return self.cti_str < other.cti_str


def generate_ctis_tlc_run_async(seed_tmpl, candidate, num_traces_per_worker=15):
    """ Starts a single instance of TLC to generate CTIs."""
    tla_ins = seed_tmpl.tla_ins
    # Avoid TLC directory naming conflicts.
    tag = random.randint(0, 10000)
    ctiseed = random.randint(0, 10000)

    # Generate spec for generating CTIs.
    invcheck_tla_indcheck = f"---- MODULE {config.specname}_CTIGen_{ctiseed} ----\n"
    invcheck_tla_indcheck += "EXTENDS %s\n\n" % config.specname

    # We shouldn't need model constants for CTI generation.
    # invcheck_tla_indcheck += self.model_consts + "\n"

    # Add definitions for for all strengthening conjuncts and for the current invariant.
    for cinvname, cinvexp in candidate.items():
        if cinvname != "Safety":
            invcheck_tla_indcheck += ("%s == %s\n" % (cinvname, cinvexp))

    # Create formula string which is conjunction of all strengthening conjuncts.
    strengthening_conjuncts_str = ""
    for cinvname in candidate.keys():
        strengthening_conjuncts_str += "    /\\ %s\n" % cinvname

    # Add definition of current inductive invariant candidate.
    invcheck_tla_indcheck += "InvStrengthened ==\n"
    # invcheck_tla_indcheck += "    /\\ Safety\n"
    invcheck_tla_indcheck += strengthening_conjuncts_str
    invcheck_tla_indcheck += "\n"

    invcheck_tla_indcheck += "IndCand ==\n"
    invcheck_tla_indcheck += "    /\\ Typeok\n"
    invcheck_tla_indcheck += "    /\\ InvStrengthened\n"

    invcheck_tla_indcheck += "===="

    indchecktlafile = f"{os.path.join(config.specdir, config.GEN_TLA_DIR)}\\{config.specname}_CTIGen_{ctiseed}.tla"
    indchecktlafilename = f"{config.GEN_TLA_DIR}\\{config.specname}_CTIGen_{ctiseed}.tla"
    f = open(indchecktlafile, 'w')
    f.write(invcheck_tla_indcheck)
    f.close()

    # Generate config file for checking inductiveness.
    os.system(f"mkdir -p {os.path.join(config.specdir, config.GEN_TLA_DIR)}")

    indcheckcfgfile = os.path.join(config.specdir, f"{config.GEN_TLA_DIR}\\{config.specname}_CTIGen_{ctiseed}.cfg")
    indcheckcfgfilename = f"{config.GEN_TLA_DIR}\\{config.specname}_CTIGen_{ctiseed}.cfg"

    invcheck_tla_indcheck_cfg = "INIT IndCand\n"
    invcheck_tla_indcheck_cfg += "NEXT Next\n"
    # invcheck_tla_indcheck_cfg += state_constraint
    invcheck_tla_indcheck_cfg += "\n"
    # Only check the invariant itself for now, and not TypeOK, since TypeOK
    # might be probabilistic, which doesn't seem to work correctly when checking
    # invariance.
    invcheck_tla_indcheck_cfg += "INVARIANT InvStrengthened\n"
    # invcheck_tla_indcheck_cfg += "INVARIANT OnePrimaryPerTerm\n"

    invcheck_tla_indcheck_cfg += seed_tmpl.constants
    invcheck_tla_indcheck_cfg += "\n"
    # if symmetry:
    #     invcheck_tla_indcheck_cfg += "SYMMETRY Symmetry\n"

    f = open(indcheckcfgfile, 'w')
    f.write(invcheck_tla_indcheck_cfg)
    f.close()

    # Use a single worker here, since we prefer to parallelize at the TLC
    # process level for probabilistic CTI generation.
    num_ctigen_tlc_workers = 6

    # # Limit simulate depth for now just to avoid very long traces.
    # simulate_flag = ""
    # if self.simulate:
    #     # traces_per_worker = self.num_simulate_traces // num_ctigen_tlc_workers
    #     traces_per_worker = num_traces_per_worker
    #     simulate_flag = "-simulate num=%d" % traces_per_worker

    logging.info(f"Using fixed TLC worker count of {num_ctigen_tlc_workers} to ensure reproducible CTI generation.")
    dirpath = tempfile.mkdtemp()

    simulate_flag = ""
    simulate_depth = 0
    if config.simulate:
        # traces_per_worker = self.num_simulate_traces // num_ctigen_tlc_workers
        traces_per_worker = num_traces_per_worker
        simulate_flag = "-simulate num=%d" % traces_per_worker
        simulate_depth = 4
    workdir = None
    if config.specdir != "":
        workdir = config.specdir

    # Apalache run.
    if config.cti_generate_use_apalache:
        # Clean the output directory
        os.system(f"rm -f {config.output_directory}/*")

        cmd = (f"java -jar {config.apalache_path} check --run-dir={config.output_directory} "
               f"--init=IndCand --inv=IndCand --length=1 --config={indcheckcfgfilename}"
               f" {indchecktlafilename}")
        logging.info("Apalache command: " + cmd)
        subproc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=workdir)
        return subproc
    else:
        cmd = (
            f'java -cp {config.TLC_PATH} tlc2.TLC'
            f' -maxSetSize {config.TLC_MAX_SET_SIZE} '
            # f' {simulate_flag} -depth {simulate_depth} '
            f' -seed {ctiseed} -noGenerateSpecTE -metadir states\\indcheckrandom_{tag}'
            f' -continue -deadlock -workers {num_ctigen_tlc_workers} -config {indcheckcfgfilename} '
            f' {indchecktlafilename}')
        logging.info("TLC command: " + cmd)

        subproc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=workdir)
        # subproc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, cwd=workdir)
        # out = subproc.stdout.decode("gbk")
        # print("out", out)
        return subproc


def generate_ctis_apalache_run_await(subproc):
    """ Awaits completion of a CTI generation process, parses its results and returns the parsed CTIs."""
    start_time = time.time()
    tlc_out = subproc.stdout.read().decode(sys.stdout.encoding)
    # print(tlc_out)
    end_time = time.time()
    # logging.debug(tlc_out)
    # lines = tlc_out.splitlines()

    all_tla_ctis = set()
    all_cti_objs = []
    outfiles = os.listdir(f"{config.output_directory}")
    for outf in outfiles:
        if "itf.json" in outf:
            # print(outf)
            cti_obj = json.load(open(f"{config.output_directory}/{outf}"))
            # print(cti_obj)
            all_cti_objs.append(cti_obj)

    for cti_obj in all_cti_objs:
        state_vars = cti_obj["vars"]
        tla_cti_str = itf_json_state_to_tla(state_vars, cti_obj["states"][0])
        # print(tla_cti_str)
        tla_cti = CTI(tla_cti_str.strip(), [], None)
        all_tla_ctis.add(tla_cti)

    # parsed_ctis = self.parse_ctis(lines)
    # return parsed_ctis
    return all_tla_ctis, end_time - start_time


def itf_json_val_to_tla_val(itfval):
    if type(itfval) == str:
        return itfval.replace("ModelValue_", "")
    if "#set" in itfval:
        return "{" + ",".join(sorted([itf_json_val_to_tla_val(v) for v in itfval["#set"]])) + "}"
    if "#tup" in itfval:
        return "<<" + ",".join([itf_json_val_to_tla_val(v) for v in itfval["#tup"]]) + ">>"


def itf_json_state_to_tla(svars, state):
    tla_state_form = ""
    for svar in svars:
        svar_line = f"/\\ {svar} = {itf_json_val_to_tla_val(state[svar])} "
        tla_state_form += svar_line
        # print(svar_line)
    # print(state)
    # print(tla_state_form)

    return tla_state_form


def parse_cti_trace(lines, curr_line):
    # Step to the 'State x' line
    # curr_line += 1
    # res = re.match(".*State (.*)\: <(.*) .*>",lines[curr_line])
    # statek = int(res.group(1))
    # action_name = res.group(2)
    # print(res)
    # print(statek, action_name)

    # print("Parsing CTI trace. Start line: " , lines[curr_line])
    # print(curr_line, len(lines))

    trace_ctis = []
    trace_action_names = []

    while curr_line < len(lines):
        if re.search('Model checking completed', lines[curr_line]):
            break

        if re.search('Error: The behavior up to this point is', lines[curr_line]):
            # print("--")
            break

        # Check for next "State N" line.
        if re.search("^State (.*)", lines[curr_line]):

            res = re.match(".*State (.*)\: <([A-Za-z0-9_-]*) .*>", lines[curr_line])
            statek = int(res.group(1))
            action_name = res.group(2)
            trace_action_names.append(action_name)
            # TODO: Consider utilizing the action for help in inferring strengthening conjuncts.
            # print(res)
            # print(statek, action_name)

            # curr_line += 1
            # print(curr_line, len(lines), lines[curr_line])
            curr_cti = ""
            curr_cti_lines = []

            # Step forward until you hit the first empty line, and add
            # each line you see as you go as a new state.
            while not lines[curr_line] == '':
                curr_line += 1
                # print("curr line", lines[curr_line])
                if len(lines[curr_line]):
                    curr_cti += (" " + lines[curr_line])

            # Save individual CTI variable lines.
            ctivars = list(filter(len, curr_cti.strip().split("/\\ ")))
            # print("varsplit:", ctivars)
            for ctivar in ctivars:
                curr_cti_lines.append("/\\ " + ctivar)

            # Assign the action names below.
            cti = CTI(curr_cti.strip(), curr_cti_lines, None)
            trace_ctis.append(cti)
            # trace_ctis.append(curr_cti.strip())
        curr_line += 1

    # Assign action names to each CTI.
    # print(trace_action_names)
    for k, cti in enumerate(trace_ctis[:-1]):
        # The action associated with a CTI is given in the state 1
        # step ahead of it in the trace.
        action_name = trace_action_names[k + 1]
        cti.setActionName(action_name)

    # for cti in trace_ctis:
    # print(cti.getActionName())

    # The last state is a bad state, not a CTI.
    trace_ctis = trace_ctis[:-1]
    return (curr_line, set(trace_ctis))


def parse_ctis(lines):
    all_ctis = set()

    curr_line = 0

    # Step forward to the first CTI error trace.
    while not re.search('Error: The behavior up to this point is', lines[curr_line]):
        curr_line += 1
        if curr_line >= len(lines):
            break

    curr_line += 1
    while curr_line < len(lines):
        (curr_line, trace_ctis) = parse_cti_trace(lines, curr_line)
        all_ctis = all_ctis.union(trace_ctis)
        curr_line += 1
    return all_ctis, 0


def generate_ctis_tlc_run_await(subproc):
    """ Awaits completion of a CTI generation process, parses its results and returns the parsed CTIs."""
    tlc_out = subproc.stdout.read().decode("gbk")
    logging.debug(tlc_out)
    lines = tlc_out.splitlines()
    if "Error: parsing" in tlc_out:
        logging.error("Error in TLC execution, printing TLC output.")
        for line in lines:
            logging.info("[TLC output] " + line)

    # Check for error:
    # 'Error: Too many possible next states for the last state in the trace'
    if "Error: Too many possible next states" in tlc_out:
        logging.error("Error in TLC execution, printing TLC output.")
        for line in lines:
            logging.info("[TLC output] " + line)
        return set()

    parsed_ctis = parse_ctis(lines)
    return parsed_ctis


def generate_ctis(seed_tmpl, candidate):
    """ Generate CTIs for use in counterexample elimination. """

    all_ctis = set()

    # Run CTI generation multiple times to gain random seed diversity. Each
    # run should call TLC with a different random seed, to generate a different
    # potential set of random initial states. We run each CTI generation process
    # in parallel, using a separate TLC instance.
    num_cti_worker_procs = 1
    cti_subprocs = []
    all_time = 0.0
    num_traces_per_tlc_instance = 15

    # Start the TLC processes for CTI generation.
    logging.info(f"Running {num_cti_worker_procs} parallel CTI generation processes")
    for n in range(num_cti_worker_procs):
        logging.info(f"Starting CTI generation process {n}")
        cti_subproc = generate_ctis_tlc_run_async(seed_tmpl=seed_tmpl,
                                                  candidate=candidate,
                                                  num_traces_per_worker=num_traces_per_tlc_instance, )

        cti_subprocs.append(cti_subproc)

    # Await completion and parse results.
    for cti_subproc in cti_subprocs:
        if config.cti_generate_use_apalache:
            parsed_ctis, times = generate_ctis_apalache_run_await(cti_subproc)
        else:
            parsed_ctis, times = generate_ctis_tlc_run_await(cti_subproc)
        all_ctis = all_ctis.union(parsed_ctis)
        all_time += times

    # FOR DIAGNOSTICS.
    # for x in sorted(list(all_ctis))[:10]:
    # print(x)
    return all_ctis, all_time
