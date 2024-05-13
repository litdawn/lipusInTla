import os


class Config:
    PT_SOLVING_TIME = 100

    SMT_CHECK_TIME = 100

    PT_SOLVING_MAX_CE = 100

    Limited_time = 100000

    cwd = os.getcwd()

    specdir = cwd + "\\Benchmarks\\Result"
    GEN_TLA_DIR = "gen_tla_dir"
    state_constraint = ""
    apalache_path = cwd + "\\apalache-0.44.2\\lib\\apalache.jar"
    jvm_args = "JVM_ARGS='-Xss16M'"
    max_num_ctis = 250
    output_directory = cwd + "\\Benchmarks\\Result\\gen_tla_dir\\apalache-cti-out"

    specname = ""

    simulate = True

    cti_generate_use_apalache = False
    TLC_PATH = ""
    TLC_MAX_SET_SIZE = 1000000

    JAVA_EXE = "java"


config = Config()
