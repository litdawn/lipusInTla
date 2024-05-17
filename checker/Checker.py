import json
import logging
import os
import random
import subprocess
import sys
import re


class Checker:
    TLC_PATH = os.path.join(os.getcwd(), "tla2tools.jar")
    TLC_MAX_SET_SIZE = 10 ** 8

    def __init__(self, spec_name, config: dict, protocols_dir, worker_num="auto", simulate_num=12500, depth=6,
                 logging_level=logging.INFO, logging_file=None):
        """
        :param spec_name: specification's name
        :param config: endive configuration content
        :param protocols_dir: to generate protocols' dir, necessay
        """
        self.worker_num = worker_num
        self.simulate_num = simulate_num
        "only use for generate ctis."
        self.depth = depth

        self.cwd = protocols_dir
        self.gen_dir = os.path.join(protocols_dir, "gen_tla_dir")
        self.state_dir = os.path.join(self.gen_dir, "states")

        self.spec_name = spec_name
        self.config = config
        if logging_file:
            logging.basicConfig(level=logging_level, filename=logging_file)
        else:
            logging.basicConfig(level=logging_level, stream=sys.stdout)
        # write default tlc cfg.
        induction_check_content = "INIT InductionInit\n"
        induction_check_content += "NEXT Next\n\n"
        induction_check_content += "INVARIANT InductionInv\n\n"
        induction_check_content += config["constants"] + "\n"
        self.induction_check_path = os.path.join(self.gen_dir, f"{spec_name}_InductionCheck.cfg")
        with open(self.induction_check_path, 'w') as f:
            f.write(induction_check_content)

    def check_invariants(self, invariants: dict):
        """
        check invariants with tlc
        :param invariants: list of lemmas to check
        :return: a dict of invariants that are satisfied within the given lemmas
        """
        # generate tla content
        seed = random.randint(0, 100000)
        tla_name = f"{self.spec_name}_InvCheck_{seed}"
        tla_content = f"---- MODULE {tla_name} ----\n"
        tla_content += f"EXTENDS {self.spec_name} \n\n"

        for inv_name, inv_content in invariants.items():
            tla_content += f"{inv_name} ==\n  {inv_content}\n"

        tla_content += "===="
        tla_path = os.path.join(self.gen_dir, f"{tla_name}.tla")
        with open(tla_path, 'w') as f:
            f.write(tla_content)

        inv_check_content = "INIT Init\n"
        inv_check_content += "NEXT Next\n\n"
        inv_check_content += "INVARIANTS " + " ".join(list(invariants.keys())) + "\n\n"
        inv_check_content += self.config["constants"] + "\n"
        inv_check_path = os.path.join(self.gen_dir, f"{self.spec_name}_InvCheck_{seed}.cfg")
        with open(inv_check_path, 'w') as f:
            f.write(inv_check_content)

        metadir = os.path.join(self.state_dir, f"inv_check_{seed}")

        cmd = (f" java -cp {Checker.TLC_PATH} tlc2.TLC -workers {self.worker_num} -deadlock -continue "
               f" -seed {seed} -metadir {metadir} -maxSetSize {Checker.TLC_MAX_SET_SIZE} -checkAllInvariants "
               f" -config {os.path.relpath(inv_check_path, self.cwd)} {os.path.relpath(tla_path, self.cwd)} ")
        logging.info(f"Check invariants with command: {cmd}")
        try:
            sub_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=self.cwd)
            output_lines = sub_process.stdout.read().decode("utf-8").splitlines()
            exit_code = sub_process.wait()
        except Exception as e:
            logging.error(f"Check invariants failed with error: {e}")
            return {}
        violated = set()
        for line in output_lines:
            res = re.match(".*Invariant (Inv_.*) is violated", line)
            if res:
                violated.add(res.group(1))
        logging.info(f"Found {len(invariants) - len(violated)} / {len(invariants)} candidate invariants satisfied")
        return {inv_name: inv_content
                for inv_name, inv_content in invariants.items()
                if inv_name not in violated}

    def check_deduction(self, deducting: dict, deducted: dict):
        """ check whether the disjunction of deducting implies deducted
        :param deducting: list of invariants to deduct
        :param deducted: invariant to be deducted
        :return: those are not deducted
        """
        # generate tla content
        seed = random.randint(0, 100000)
        tla_name = f"{self.spec_name}_DeductionCheck_{seed}"
        tla_content = f"---- MODULE {tla_name} ----\n"
        tla_content += f"EXTENDS {self.spec_name} \n\n"

        disjunction = ""
        for name, content in deducting.items():
            tla_content += f"{name} ==\n  {content}\n"
            disjunction += f"  /\\ {name}\n"
        tla_content += "\n"
        tla_content += (f"Deducting ==\n"
                        f"  /\\ {self.config['typeok']}\n"
                        f"  /\\ {self.config['safety']}\n")

        tla_content += disjunction + "\n"
        for name, content in deducted.items():
            tla_content += (f"{name} == \n"
                            f"  {content}\n")

        tla_content += "===="
        tla_path = os.path.join(self.gen_dir, f"{tla_name}.tla")
        with open(tla_path, 'w') as f:
            f.write(tla_content)

        deduction_check_content = "INIT Deducting\n"
        deduction_check_content += "NEXT Next\n\n"
        deduction_check_content += "INVARIANTS " + " ".join(deducted.keys()) + "\n\n"
        deduction_check_content += self.config["constants"] + "\n"
        deduction_check_path = os.path.join(self.gen_dir, f"{self.spec_name}_DeductionCheck_{seed}.cfg")
        with open(deduction_check_path, 'w') as f:
            f.write(deduction_check_content)

        metadir = os.path.join(self.state_dir, f"deduction_check_{seed}")

        cmd = (f" java -cp {Checker.TLC_PATH} tlc2.TLC -workers {self.worker_num} -deadlock -continue "
               f" -seed {seed} -metadir {metadir} -maxSetSize {Checker.TLC_MAX_SET_SIZE} -checkAllInvariants "
               f" -config {os.path.relpath(deduction_check_path, self.cwd)} {os.path.relpath(tla_path, self.cwd)} ")

        logging.info(f"Check deduction with command: {cmd}")
        try:
            sub_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=self.cwd)
            output_lines = sub_process.stdout.read().decode("utf-8").splitlines()
            exit_code = sub_process.wait()
        except Exception as e:
            logging.error(f"Check deduction failed with error: {e}")
            return set()
        not_deducted = set()
        for line in output_lines:
            res = re.match(".*Invariant (Inv_.*) is violated", line)
            if res:
                not_deducted.add(res.group(1))
        logging.info(f"Found {len(not_deducted)} / {len(deducted)} candidate invariants not deducted")
        return {name: deducted[name] for name in not_deducted}

    def check_induction(self, ind_lemmas: dict):
        """
        check induction with tlc
        :param ind_lemmas: list of lemmas' disjunction to check
        :return: whether is inductive, ctis when not inductive
        """
        seed = random.randint(0, 100000)
        tla_name = f"{self.spec_name}_InductionCheck_{seed}"
        tla_content = f"---- MODULE {tla_name} ----\n"
        tla_content += f"EXTENDS {self.spec_name} \n\n"

        for name, content in ind_lemmas.items():
            tla_content += f"{name} ==\n  {content}\n"
        tla_content += "\n"

        induction_invariant = f"  /\\ {self.config['safety']} \n"
        for induction_lemma_name in ind_lemmas.keys():
            induction_invariant += f"  /\\ {induction_lemma_name}\n"
        tla_content += f"InductionInv ==\n{induction_invariant}\n"
        tla_content += f"InductionInit ==\n  /\\ {self.config['typeok']}\n"
        tla_content += induction_invariant + "\n"
        tla_content += "===="
        tla_path = os.path.join(self.gen_dir, f"{tla_name}.tla")
        with open(tla_path, 'w') as f:
            f.write(tla_content)

        metadir = os.path.join(self.state_dir, f"induction_check_{seed}")
        cmd = (f" java -cp {Checker.TLC_PATH} tlc2.TLC -workers {self.worker_num} -deadlock -continue "
               f" -seed {seed} -metadir {metadir} -maxSetSize {Checker.TLC_MAX_SET_SIZE}  "
               f" -config {os.path.relpath(self.induction_check_path, self.cwd)} {os.path.relpath(tla_path, self.cwd)} ")
        logging.info(f"Check induction with command: {cmd}")
        try:
            sub_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=self.cwd)
            output = sub_process.stdout.read().decode("utf-8")
            exit_code = sub_process.wait()
        except Exception as e:
            logging.error(f"Check induction failed with error: {e}")
            return False, set()
        ctis = CTI.parse_ctis(output.splitlines())
        if len(ctis) > 0:
            logging.info(f"Disjunction is not inductive, found {len(ctis)} CTIs")
            return False, ctis
        return True, set()

    def generate_cti(self, lemma_invs: dict):
        """
        generate CTIs with tlc, based on the given invariants
        :param lemma_invs: given lemmas
        :return:
        """
        invariants = lemma_invs.values()
        cti_gen_init = f"CTIGenInit ==\n  /\\ {self.config['typeok']}\n  /\\ {self.config['safety']}\n"
        for inv in invariants:
            cti_gen_init += f"  /\\ {inv}\n"
        seed = random.randint(0, 100000)
        tla_name = f"{self.spec_name}_CTIGen_{seed}"
        tla_content = f"---- MODULE {tla_name} ----\n"
        tla_content += f"EXTENDS {self.spec_name} \n\n"
        tla_content += f"{cti_gen_init}\n\n"
        tla_content += "===="
        tla_path = os.path.join(self.gen_dir, f"{tla_name}.tla")
        with open(tla_path, 'w') as f:
            f.write(tla_content)

        cfg_content = f"INIT CTIGenInit\n"
        cfg_content += f"NEXT Next\n\n"
        cfg_content += f"INVARIANT {self.config['safety']}\n\n"
        cfg_content += self.config["constants"] + "\n"
        cfg_path = os.path.join(self.gen_dir, f"{self.spec_name}_CTIGen_{seed}.cfg")
        with open(cfg_path, 'w') as f:
            f.write(cfg_content)

        cmd = (f" java -cp {Checker.TLC_PATH} tlc2.TLC -workers {self.worker_num} -deadlock -continue "
               f" -simulate num={self.simulate_num} -depth {self.depth} "
               f" -seed {seed} -maxSetSize {Checker.TLC_MAX_SET_SIZE} "
               f" -config {os.path.relpath(cfg_path, self.cwd)} {os.path.relpath(tla_path, self.cwd)} ")
        logging.info(f"Generate CTIs with command: {cmd}")
        try:
            sub_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=self.cwd)
            output = sub_process.stdout.read().decode("utf-8")
            exit_code = sub_process.wait()
        except Exception as e:
            logging.error(f"Generate CTIs failed with error: {e}")
            return set()
        ctis = CTI.parse_ctis(output.splitlines())
        logging.info(f"Found {len(ctis)} CTIs")
        return ctis

    def eliminate_ctis(self, add_invs: dict, lemmas: dict, ctis: set):
        """
        eliminate CTIs with tlc, based on the given lemmas
        :param add_invs: invariants those added to induction invariants
        :param lemmas: new given lemmas
        :param ctis: counterexamples to induction to eliminate
        :return: dict of which lemma kill which ctis in type dict
        """
        cti_eliminated_by_lemmas = {}
        for name in lemmas.keys():
            cti_eliminated_by_lemmas[name] = set()
        if not ctis:
            return cti_eliminated_by_lemmas
        cti_table = {}
        for cti in ctis:
            hashed = str(hash(cti))
            cti_table[hashed] = cti

        seed = random.randint(0, 100000)
        # generate tla content
        tla_name = f"{self.spec_name}_CTIEliminate_{seed}"
        tla_content = f"---- MODULE {tla_name} ----\n"
        tla_content += f"EXTENDS {self.spec_name}, Naturals, TLC\n\n"
        tla_content += self.config['model_consts'] + "\n"
        invars = ""
        inv_vals = "InvVals ==\n  /\\ TRUE\n"
        cti_check_next = "CTICheckNext ==\n  /\\ NextUnchanged\n  /\\ UNCHANGED ctiId\n"
        for name, content in lemmas.items():
            tla_content += f"VARIABLE {name}_val\n"
            invars += f"{name} == {content}\n"
            inv_vals += f"  /\\ {name}_val = {name}\n"
            cti_check_next += f"  /\\ UNCHANGED {name}_val"

        tla_content += "VARIABLE ctiId\n\n"
        for name, content in add_invs.items():
            tla_content += f"{name} == {content}\n"

        tla_content += "\n" + invars + "\n"
        tla_content += "kCTIs ==\n"
        for hash_code, cti in cti_table.items():
            tla_content += f"\t\\/ {cti.getCTIStateString()}\n"
            tla_content += f"\t   /\\ ctiId = \"{hash_code}\"\n"
        tla_content += "\n"
        tla_content += inv_vals + "\n"

        tla_content += ("CTICheckInit ==\n"
                        "  /\\ kCTIs\n"
                        "  /\\ InvVals\n")
        for name in add_invs.keys():
            tla_content += f"  /\\ {name}\n"
        tla_content += "\n"

        tla_content += cti_check_next + "\n"
        tla_content += "===="
        tla_path = os.path.join(self.gen_dir, f"{tla_name}.tla")
        with open(tla_path, 'w') as f:
            f.write(tla_content)
        # generate cfg content
        cfg_content = "INIT CTICheckInit\n"
        cfg_content += "NEXT CTICheckNext\n\n"
        cfg_content += self.config["constants"] + "\n"
        cfg_path = os.path.join(self.gen_dir, f"{tla_name}.cfg")
        with open(cfg_path, 'w') as f:
            f.write(cfg_content)
        json_path = os.path.join(self.state_dir, f"{tla_name}.json")
        metadir = os.path.join(self.state_dir, f"{tla_name}_metadir")

        cmd = (f" java -cp {Checker.TLC_PATH} tlc2.TLC -workers {self.worker_num} "
               f" -maxSetSize {Checker.TLC_MAX_SET_SIZE} -deadlock -continue "
               f" -dump json {json_path} -noGenerateSpecTE -metadir {metadir} -checkAllInvariants  "
               f" -config {os.path.relpath(cfg_path, self.cwd)} {os.path.relpath(tla_path, self.cwd)} ")
        logging.info(f"Eliminate CTIs with command: {cmd}")
        try:
            sub_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=self.cwd)
            output = sub_process.stdout.read().decode("utf-8")
            exit_code = sub_process.wait()
            if 'Error: ' in output or 'error: ' in output:
                raise Exception(f"Eliminate CTIs failed with error: {output}")
        except Exception as e:
            logging.error(f"Eliminate CTIs failed with error: {e}")
            return cti_eliminated_by_lemmas
        with open(json_path, 'r') as f:
            cti_states = json.load(f)['states']

        for state in cti_states:
            vals = state['val']
            cti_id = vals['ctiId']
            for name in lemmas.keys():
                if not vals[f"{name}_val"]:
                    cti_eliminated_by_lemmas[name].add(cti_id)

        cti_eliminated_by_lemmas = dict(sorted(cti_eliminated_by_lemmas.items(), key=lambda x: len(x[1])))
        eliminated_ctis = set()
        for cti_ids in cti_eliminated_by_lemmas.values():
            [eliminated_ctis.add(cti_id) for cti_id in cti_ids]
        logging.info(f"Eliminated {len(eliminated_ctis)} / {len(ctis)} CTIs in this round")
        return cti_eliminated_by_lemmas


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

    @staticmethod
    def parse_cti_trace(output_lines, curr_line):
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

        while curr_line < len(output_lines):
            if re.search('Model checking completed', output_lines[curr_line]):
                break

            if re.search('Error: The behavior up to this point is', output_lines[curr_line]):
                # print("--")
                break

            # Check for next "State N" line.
            if re.search("^State (.*)", output_lines[curr_line]):

                res = re.match(".*State (.*): <([A-Za-z0-9_-]*) .*>", output_lines[curr_line])
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
                while not output_lines[curr_line] == '':
                    curr_line += 1
                    # print("curr line", lines[curr_line])
                    if len(output_lines[curr_line]):
                        curr_cti += (" " + output_lines[curr_line])

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

    @staticmethod
    def parse_ctis(output_lines):
        """
        Parse all CTIs from the output of TLC.
        :param output_lines: tlc output lines
        :return: ctis: set of CTIs
        """
        all_ctis = set()

        curr_line = 0

        # Step forward to the first CTI error trace.
        while not re.search('Error: The behavior up to this point is', output_lines[curr_line]):
            curr_line += 1
            if curr_line >= len(output_lines):
                break

        curr_line += 1
        while curr_line < len(output_lines):
            (curr_line, trace_ctis) = CTI.parse_cti_trace(output_lines, curr_line)
            all_ctis = all_ctis.union(trace_ctis)
            curr_line += 1
        return all_ctis
