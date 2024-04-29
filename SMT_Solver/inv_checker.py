import random
import logging
import sys
import os
import time
import subprocess
import re
import json
import argparse
import platform
from datetime import datetime
import tempfile

TLC_MAX_SET_SIZE = 10 ** 8
JAVA_EXE = "java"

# todo 此处仍需理通逻辑
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


def check_invariants(invs: list, specname, tla_ins, strengthening_conjuncts="", tlc_workers=6):
    seed = time.time()
    """ Check which of the given invariants are valid. """
    ta = time.time()
    invcheck_tla = "---- MODULE %s_InvCheck_%d ----\n" % (specname, seed)
    invcheck_tla += "EXTENDS %s\n\n" % specname
    invcheck_tla += self.model_consts + "\n"

    all_inv_names = set()
    for i, inv in enumerate(invs):
        sinv = ("Inv%d == " % i) + self.quant_inv(inv)
        all_inv_names.add("Inv%d" % i)
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

    if type(tla_ins.constants) == list:
        constants = "\n".join(tla_ins.constants)
    else:
        constants = tla_ins.constants

    invcheck_cfg += constants
    invcheck_cfg += "\n"
    invcheck_cfg += state_constraint
    invcheck_cfg += "\n"
    # if self.symmetry:
    #     invcheck_cfg += "SYMMETRY Symmetry\n"

    for invi in range(len(invs)):
        sinv = "INVARIANT Inv" + str(invi)
        invcheck_cfg += sinv + "\n"

    invcheck_cfg_file = f"{os.path.join(specdir, GEN_TLA_DIR)}/{specname}_InvCheck_{seed}.cfg"
    invcheck_cfg_filename = f"{GEN_TLA_DIR}/{specname}_InvCheck_{seed}.cfg"

    f = open(invcheck_cfg_file, 'w')
    f.write(invcheck_cfg)
    f.close()

    # Check invariants.
    logging.info("Checking %d candidate invariants in spec file '%s'" % (len(invs), invcheck_spec_name))
    workdir = None if specdir == "" else specdir
    violated_invs = runtlc_check_violated_invariants(invcheck_spec_name, config=invcheck_cfg_filename,
                                                     tlc_workers=tlc_workers, cwd=workdir, java=java_exe)
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
    dirpath = tempfile.mkdtemp()
    metadir_path = f"states/states_{random.randint(0, 1000000000)}"
    cmd = java + (
        f' -Djava.io.tmpdir="{dirpath}" -cp tla2tools-checkall.jar tlc2.TLC {tlc_flags} -maxSetSize {TLC_MAX_SET_SIZE} -metadir {metadir_path} -noGenerateSpecTE -checkAllInvariants -deadlock -continue -workers {tlc_workers}')
    if config:
        cmd += " -config " + config
    cmd += " " + spec
    logging.info("TLC command: " + cmd)
    subproc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=cwd)
    tlc_raw_out = ""
    line_str = ""
    tlc_lines = []
    while True:
        new_stdout = subproc.stdout.read(1).decode(sys.stdout.encoding)
        if new_stdout == "":  # reached end of file.
            break
        if new_stdout == os.linesep:
            logging.debug("[TLC Output] " + line_str)
            tlc_lines.append(line_str)
            line_str = ""
        else:
            line_str += new_stdout
    return tlc_lines
