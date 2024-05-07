import random
import logging
import os
import time
import subprocess
import sys
from SMT_Solver.Util import *
from SMT_Solver.Config import config
import tempfile

logging.basicConfig(level=logging.INFO)


def check_invariants(invs: list, seed_tmpl, strengthening_conjuncts="", tlc_workers=3):
    seed = random.randint(0, 10000)
    """ Check which of the given invariants are valid. """
    ta = time.time()
    invcheck_tla = "---- MODULE %s_InvCheck_%d ----\n" % (config.specname, seed)
    invcheck_tla += "EXTENDS %s\n\n" % config.specname

    # invcheck_tla += tla_ins.model_const + "\n"
    invcheck_tla += seed_tmpl.model_consts + "\n"

    all_inv_names = set()
    for i, inv in enumerate(invs):
        sinv = inv
        all_inv_names.add(inv.split("==")[0].strip())
        invcheck_tla += sinv + "\n"

    invcheck_tla += "===="

    invcheck_spec_name = f"{config.GEN_TLA_DIR}\\{config.specname}_InvCheck_{seed}"
    invcheck_spec_filename = f"{os.path.join(config.specdir, config.GEN_TLA_DIR)}\\{config.specname}_InvCheck_{seed}"
    invchecktlafile = invcheck_spec_filename + ".tla"
    f = open(invchecktlafile, 'w')
    f.write(invcheck_tla)
    f.close()

    invcheck_cfg = "INIT Init\n"
    invcheck_cfg += "NEXT Next\n"

    invcheck_cfg += seed_tmpl.constants
    invcheck_cfg += "\n"
    # invcheck_cfg += state_constraint
    invcheck_cfg += "\n"
    # if self.symmetry:
    #     invcheck_cfg += "SYMMETRY Symmetry\n"

    for i, inv_content in enumerate(invs):
        real_content = inv_content.split("==")[0].strip()
        sinv = f"INVARIANT {real_content}"
        invcheck_cfg += sinv + "\n"

    invcheck_cfg_file = f"{config.specdir}\\{config.GEN_TLA_DIR}\\{config.specname}_InvCheck_{seed}.cfg"
    invcheck_cfg_file = invcheck_cfg_file.replace("/", "\\")
    invcheck_cfg_filename = f"{config.GEN_TLA_DIR}\\{config.specname}_InvCheck_{seed}.cfg"

    f = open(invcheck_cfg_file, 'w')
    f.write(invcheck_cfg)
    f.close()

    # Check invariants.
    logging.info("Checking %d candidate invariants in spec file '%s'" % (len(invs), invcheck_spec_name))
    workdir = None if config.specdir == "" else config.specdir
    violated_invs = runtlc_check_violated_invariants(invchecktlafile, cfg=invcheck_cfg_file,
                                                     tlc_workers=tlc_workers, cwd=workdir, java=config.JAVA_EXE)
    sat_invs = (all_inv_names - violated_invs)
    logging.info(
        f"Found {len(sat_invs)} / {len(invs)} candidate invariants satisfied in {round(time.time() - ta, 2)}s.")

    return sat_invs


def runtlc_check_violated_invariants(spec, cfg=None, tlc_workers=6, cwd=None, java="java"):
    #
    # TODO: Check for this type of error:
    # 'Error: The invariant of Inv91 is equal to FALSE'
    #
    lines = run_tlc(spec, cfg=cfg, tlc_workers=tlc_workers, cwd=cwd)
    # print("lines", lines)
    invs_violated = set()
    for l in grep_lines("is violated", lines):
        res = re.match(".*Invariant (inv_.*) is violated", l)
        inv_name = res.group(1)
        invs_violated.add(inv_name)
    return invs_violated


def run_tlc(spec, cfg=None, tlc_workers=6, cwd=None, tlc_flags=""):
    # Make a best effort to attempt to avoid collisions between different
    # instances of TLC running on the same machine.
    dirpath = tempfile.mkdtemp()
    metadir_path = f"states\\states_{random.randint(0, 1000000000)}"

    cmd = (
        # ' -metadir {metadir_path} -noGenerateSpecTE '
        f'java -Djava.io.tmpdir="{dirpath}" -cp {config.TLC_PATH} tlc2.TLC  {tlc_flags} -maxSetSize {config.TLC_MAX_SET_SIZE}'
        f' -checkAllInvariants -metadir {metadir_path} -noGenerateSpecTE -deadlock -continue -workers {tlc_workers}'
    )
    if config:
        cmd += " -config " + cfg
    cmd += " " + spec
    logging.info("TLC command: " + cmd)
    subproc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)

    line_str = ""
    tlc_lines = []
    new_stdout = subproc.stdout.decode(sys.stdout.encoding)
    tlc_lines = new_stdout.split("\r\n")
    # while True:
    #     new_stdout = subproc.stdout.read(1).decode(sys.stdout.encoding)
    #
    #     if new_stdout == "":  # reached end of file.
    #         break
    #     if new_stdout == os.linesep:
    #         logging.info("[TLC Output] " + line_str)
    #         tlc_lines.append(line_str)
    #         line_str = ""
    #     else:
    #         line_str += new_stdout
    # print(f"tlc_lines: {tlc_lines}")
    return tlc_lines
