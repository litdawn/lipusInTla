import random
import os
import logging
import json
import sys
import subprocess
import time

# 输入的tla/cfg在
# 新tla: specdir//gen_tla_dir/{specname}_CTIGen_{ctiseed}.tla
# 新cfg：specdir//gen_tla_dir/{specname}_CTIGen_{ctiseed}.cfg
# 生成的cti文件在 Benchmarks/gen_tla/apalache-cti-ou
specdir = os.getcwd() + "\\Result"
GEN_TLA_DIR = "gen_tla"
state_constraint = ""
apalache_path = os.getcwd() + "\\apalache-0.44.2\\lib\\apalache.jar"
jvm_args = "JVM_ARGS='-Xss16M'"
max_num_ctis = 250
output_directory = os.getcwd() + "Benchmarks/gen_tla/apalache-cti-out"


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


def generate_ctis_tlc_run_async(specname, tla_ins, strengthening_conjuncts="", num_traces_per_worker=15):
    """ Starts a single instance of TLC to generate CTIs."""

    # Avoid TLC directory naming conflicts.
    # tag = random.randint(0, 10000)
    ctiseed = random.randint(0, 10000)

    # Generate spec for generating CTIs.
    invcheck_tla_indcheck = f"---- MODULE {specname}_CTIGen_{ctiseed} ----\n"
    invcheck_tla_indcheck += "EXTENDS %s\n\n" % specname

    # We shouldn't need model constants for CTI generation.
    # invcheck_tla_indcheck += self.model_consts + "\n"

    # Add definitions for for all strengthening conjuncts and for the current invariant.
    for cinvname, cinvexp in strengthening_conjuncts:
        invcheck_tla_indcheck += ("%s == %s\n" % (cinvname, cinvexp))

    # Create formula string which is conjunction of all strengthening conjuncts.
    strengthening_conjuncts_str = ""
    for cinvname, cinvexp in strengthening_conjuncts:
        strengthening_conjuncts_str += "    /\\ %s\n" % cinvname

    # Add definition of current inductive invariant candidate.
    invcheck_tla_indcheck += "InvStrengthened ==\n"
    invcheck_tla_indcheck += "    /\\ Safety\n"
    invcheck_tla_indcheck += strengthening_conjuncts_str
    invcheck_tla_indcheck += "\n"

    invcheck_tla_indcheck += "IndCand ==\n"
    invcheck_tla_indcheck += "    /\\ Typeok\n"
    invcheck_tla_indcheck += "    /\\ InvStrengthened\n"

    invcheck_tla_indcheck += "===="

    indchecktlafile = f"{os.path.join(specdir, GEN_TLA_DIR)}/{specname}_CTIGen_{ctiseed}.tla"
    indchecktlafilename = f"{GEN_TLA_DIR}/{specname}_CTIGen_{ctiseed}.tla"
    f = open(indchecktlafile, 'w')
    f.write(invcheck_tla_indcheck)
    f.close()

    # Generate config file for checking inductiveness.
    os.system(f"mkdir -p {os.path.join(specdir, GEN_TLA_DIR)}")

    indcheckcfgfile = os.path.join(specdir, f"{GEN_TLA_DIR}/{specname}_CTIGen_{ctiseed}.cfg")
    indcheckcfgfilename = f"{GEN_TLA_DIR}/{specname}_CTIGen_{ctiseed}.cfg"

    invcheck_tla_indcheck_cfg = "INIT IndCand\n"
    invcheck_tla_indcheck_cfg += "NEXT Next\n"
    invcheck_tla_indcheck_cfg += state_constraint
    invcheck_tla_indcheck_cfg += "\n"
    # Only check the invariant itself for now, and not TypeOK, since TypeOK
    # might be probabilistic, which doesn't seem to work correctly when checking
    # invariance.
    invcheck_tla_indcheck_cfg += "INVARIANT InvStrengthened\n"
    # invcheck_tla_indcheck_cfg += "INVARIANT OnePrimaryPerTerm\n"
    if type(tla_ins.constants) == list:
        constants = "\n".join(tla_ins.constants)
    else:
        constants = tla_ins.constants
    invcheck_tla_indcheck_cfg += constants
    invcheck_tla_indcheck_cfg += "\n"
    # if symmetry:
    #     invcheck_tla_indcheck_cfg += "SYMMETRY Symmetry\n"

    f = open(indcheckcfgfile, 'w')
    f.write(invcheck_tla_indcheck_cfg)
    f.close()

    # Use a single worker here, since we prefer to parallelize at the TLC
    # process level for probabilistic CTI generation.
    num_ctigen_tlc_workers = 1

    # # Limit simulate depth for now just to avoid very long traces.
    # simulate_flag = ""
    # if self.simulate:
    #     # traces_per_worker = self.num_simulate_traces // num_ctigen_tlc_workers
    #     traces_per_worker = num_traces_per_worker
    #     simulate_flag = "-simulate num=%d" % traces_per_worker

    logging.info(f"Using fixed TLC worker count of {num_ctigen_tlc_workers} to ensure reproducible CTI generation.")
    # dirpath = tempfile.mkdtemp()

    # Apalache run.
    # if self.use_apalache_ctigen:
    # Clean the output directory.
    os.system(f"rm -f {output_directory}/*")

    cmd = f"java.exe -jar {apalache_path} check --run-dir={output_directory} --max-error={max_num_ctis} --init=IndCand --inv=IndCand --length=1 --config={indcheckcfgfilename} {indchecktlafilename}"
    # cmd = self.java_exe + ' -Djava.io.tmpdir="%s" -cp tla2tools-checkall.jar tlc2.TLC -maxSetSize %d %s -depth %d -seed %d -noGenerateSpecTE -metadir states/indcheckrandom_%d -continue -deadlock -workers %d -config %s %s' % args
    logging.info("Apalache command: " + cmd)
    workdir = None
    if specdir != "":
        workdir = specdir
    subproc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=workdir)

    return subproc


def generate_ctis_apalache_run_await(subproc):
    """ Awaits completion of a CTI generation process, parses its results and returns the parsed CTIs."""
    start_time = time.time()
    tlc_out = subproc.stdout.read().decode(sys.stdout.encoding)
    end_time = time.time()
    # logging.debug(tlc_out)
    # lines = tlc_out.splitlines()

    all_tla_ctis = set()
    all_cti_objs = []
    outfiles = os.listdir(f"{output_directory}")
    for outf in outfiles:
        if "itf.json" in outf:
            # print(outf)
            cti_obj = json.load(open(f"{output_directory}/{outf}"))
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
    return all_tla_ctis, end_time-start_time


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


def generate_ctis(tla_ins, path2cfg):
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
        cti_subproc = generate_ctis_tlc_run_async(specname=path2cfg.split("//")[-1].split(".")[0],
                                                  tla_ins=tla_ins,
                                                  num_traces_per_worker=num_traces_per_tlc_instance, )

        cti_subprocs.append(cti_subproc)

    # Await completion and parse results.
    for cti_subproc in cti_subprocs:
        parsed_ctis, times = generate_ctis_apalache_run_await(cti_subproc)
        all_ctis = all_ctis.union(parsed_ctis)
        all_time += times

    # FOR DIAGNOSTICS.
    # for x in sorted(list(all_ctis))[:10]:
    # print(x)
    return all_ctis, all_time
