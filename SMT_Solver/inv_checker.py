import random
import logging
import sys
import os
import time
import subprocess
import re
import tempfile

logging.basicConfig(level=logging.INFO)

TLC_MAX_SET_SIZE = 10 ** 8
JAVA_EXE = "java"

# todo 此处仍需理通逻辑
# 输入的tla/cfg在
# 新tla: specdir//gen_tla_dir/{specname}_CTIGen_{ctiseed}.tla
# 新cfg：specdir//gen_tla_dir/{specname}_CTIGen_{ctiseed}.cfg
# 生成的cti文件在 Benchmarks/gen_tla/apalache-cti-ou
specdir = os.getcwd() + "/Result"
GEN_TLA_DIR = "gen_tla_dir"
state_constraint = ""
apalache_path = os.getcwd() + "/apalache-0.44.2/lib/apalache.jar"
jvm_args = "JVM_ARGS='-Xss16M'"
max_num_ctis = 250
output_directory = os.getcwd() + "Benchmarks/gen_tla/apalache-cti-out"


def check_invariants(invs: list, specname, tla_ins, strengthening_conjuncts="", tlc_workers=6):
    seed = random.randint(0,10000)
    """ Check which of the given invariants are valid. """
    ta = time.time()
    invcheck_tla = "---- MODULE %s_InvCheck_%d ----\n" % (specname, seed)
    invcheck_tla += "EXTENDS %s\n\n" % specname

    # invcheck_tla += tla_ins.model_const + "\n"
    invcheck_tla += "CONSTANT "+ "".join(tla_ins.constants.keys()) +"\n"

    all_inv_names = set()
    for i, inv in enumerate(invs):
        sinv = inv
        all_inv_names.add(inv.split("==")[0])
        invcheck_tla += sinv + "\n"

    invcheck_tla += "===="

    invcheck_spec_name = f"{GEN_TLA_DIR}/{specname}_InvCheck_{seed}"
    invcheck_spec_filename = f"{os.path.join(specdir, GEN_TLA_DIR)}/{specname}_InvCheck_{seed}"
    invchecktlafile = invcheck_spec_filename + ".tla"
    f = open(invchecktlafile, 'w')
    f.write(invcheck_tla)
    f.close()

    invcheck_cfg = "INIT Init\n"
    invcheck_cfg += "NEXT Next\n"


    invcheck_cfg += tla_ins.model_const
    invcheck_cfg += "\n"
    invcheck_cfg += state_constraint
    invcheck_cfg += "\n"
    # if self.symmetry:
    #     invcheck_cfg += "SYMMETRY Symmetry\n"

    for i,inv_content in enumerate(invs):
        real_content = inv_content.split("==")[0].strip()
        sinv = f"INVARIANT {real_content}"
        invcheck_cfg += sinv + "\n"

    invcheck_cfg_file = f"{os.path.join(specdir, GEN_TLA_DIR)}/{specname}_InvCheck_{seed}.cfg"
    invcheck_cfg_filename = f"{GEN_TLA_DIR}/{specname}_InvCheck_{seed}.cfg"

    f = open(invcheck_cfg_file, 'w')
    f.write(invcheck_cfg)
    f.close()

    # Check invariants.
    logging.info("Checking %d candidate invariants in spec file '%s'" % (len(invs), invcheck_spec_name))
    workdir = None if specdir == "" else specdir
    violated_invs = runtlc_check_violated_invariants(invcheck_spec_filename, config=invcheck_cfg_file,
                                                     tlc_workers=tlc_workers, cwd=workdir, java=JAVA_EXE)
    sat_invs = (all_inv_names - violated_invs)
    logging.info(
        f"Found {len(sat_invs)} / {len(invs)} candidate invariants satisfied in {round(time.time() - ta, 2)}s.")

    return sat_invs


def grep_lines(pattern, lines):
    return [ln for ln in lines if re.search(pattern, ln)]


def runtlc_check_violated_invariants(spec, config=None, tlc_workers=6, cwd=None, java="java"):
    #
    # TODO: Check for this type of error:
    # 'Error: The invariant of Inv91 is equal to FALSE'
    #
    lines = run_tlc(spec, config=config, tlc_workers=tlc_workers, cwd=cwd, java=java)
    invs_violated = set()
    for l in grep_lines("is violated", lines):
        res = re.match(".*Invariant (Inv.*) is violated", l)
        inv_name = res.group(1)
        invs_violated.add(inv_name)
    return invs_violated


def run_tlc(spec, config=None, tlc_workers=6, cwd=None, java="java", tlc_flags=""):
    # Make a best effort to attempt to avoid collisions between different
    # instances of TLC running on the same machine.
    # dirpath = tempfile.mkdtemp()
    metadir_path = f"states/states_{random.randint(0, 1000000000)}"
    from main import TLC_PATH
    cmd = (
        # ' -metadir {metadir_path} -noGenerateSpecTE '
        f' java -cp {TLC_PATH} tlc2.TLC  -maxSetSize {TLC_MAX_SET_SIZE}'
        f' -checkAllInvariants -deadlock -continue -workers {tlc_workers}')
    if config:
        cmd += " -config " + config
    cmd += " " + spec + ".tla"
    logging.info("TLC command: " + cmd)
    subproc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    line_str = ""
    tlc_lines = []
    # new_stdout = subproc.stdout.read(1).decode(sys.stdout.encoding)
    new_stdout = subproc.stderr.decode("gbk")
    print("new_stderr", new_stdout)
    if new_stdout == "":  # reached end of file.
        return []
    if new_stdout == os.linesep:
        logging.info("[TLC Output] " + line_str)
        tlc_lines.append(line_str)
        line_str = ""
    else:
        line_str += new_stdout
    return tlc_lines
