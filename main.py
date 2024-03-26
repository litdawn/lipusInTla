# A pipeline framework to realize the RL Pruning Tool for loop invariant inference
import argparse
import subprocess
import time

# from PT_generators.RL_Prunning.PT_generator import PT_generator
from PT_generators.SimplestIteration.PT_generator import PT_generator
from SMT_Solver import Template_solver
from SMT_Solver.Config import config
from SMT_Solver.SMT_verifier import SMT_verifier
from Utilities.ArgParser import parseArgs



def main(path2tla):
    start_time = time.time()
    # Step 1. Input the three formation of the code.
    #path2CFile, path2CFG, path2SMT = parseArgs()
    # Step 2. Load the Partial Template Generator.
    pT_generator = PT_generator(path2tla)
    sMT_verifier = SMT_verifier(pT_generator.var_names)
    # Step 3. ENTER the ICE Solving Loop
    solved = False
    CE = {'p': [],
          'n': [],
          'i': []}
    print("Begin_process:   ", path2tla)
    Iteration = 0
    while not solved:
        current_time = time.time()
        if current_time - start_time >= config.Limited_time:
            print("Loop invariant Inference is OOT")
            return None,None
        Iteration += 1
        # Step 3.1 Generate A partial template
        PT = pT_generator.generate_next(CE)
        if PT is None:
            print("The only way is to give up now")
            return None,None
        # Step 3.2 Solving the partial template
        try:
            Can_I = Template_solver.solve(PT, CE)
            print(Can_I)
            #raise TimeoutError # try this thing out
        except TimeoutError as OOT:  # Out Of Time, we punish
            pT_generator.punish('STRICT', 'VERY', 'S')
            continue
        if Can_I is None:  # Specified too much, we loose.
            pT_generator.punish('LOOSE', 'MEDIUM', 'S')
            continue
        # Step 3.3 Check if we bingo
        try:
            Counter_example = sMT_verifier.verify(Can_I, path2tla)
        except TimeoutError as OOT:  # Out Of Time, we punish
            pT_generator.punish('STRICT', 'LITTLE', 'V')
            continue
        if Counter_example is None:  # Bingo, we prise
            solved = True
            print("The answer is :  ", str(Can_I))
            pT_generator.prise('VERY')
            current_time = time.time()
            print("Time cost is :  ", str(current_time - start_time))
            return current_time - start_time, str(Can_I)
        else:  # progressed anyway, we prise
            if Counter_example.assignment not in CE[Counter_example.kind]:
                CE[Counter_example.kind].append(Counter_example.assignment)
            pT_generator.prise('LITTLE')
            continue
if __name__ == "__main__":
    path2tla=r"Benchmarks/tla/firewall.tla"
    # path2config = path2tla[:-3] + "cfg"
    # command = f"java.exe -jar apalache-0.44.2/lib/apalache.jar check --inv=Inv --run-dir=gen_tla/apalache-cti-out --config={path2config} {path2tla} "
    # result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print(result)
    # path2CFG=r"Benchmarks/Linear/c_graph/2.c.json"
    # path2SMT=r"Benchmarks/Linear/c_smt2/2.c.smt"
    main(path2tla)