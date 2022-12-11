import sys
import subprocess
import argparse

import numpy as np
import os
from os import listdir
import shutil
import csv
from Bio.PDB.PDBParser import PDBParser
sys.path.insert(0, '/cs/labs/dina/seanco/xl_parser')
import general_utils
import alpha_fold_files
sys.path.insert(0, '/cs/labs/dina/seanco/xl_parser/alphafold')
import collabfold_script_moriah

XL_DOCKING_SCORE_EXE = "/cs/labs/dina/seanco/DockingXlScore/cmake-build-debug/DockingXlScore"
PDB_2_FASTA = '/cs/staff/dina/utils/pdb2fasta'
FASTA_PATH = '/cs/labs/dina/seanco/xl_parser/fasta_files/'
RUN_ALPHAFOLD_SCRIPT = '/cs/labs/dina/seanco/xl_parser/alphafold/collabfold_script_moriah.py'
ALIGN_RMSD_SCRIPT = '/cs/staff/dina/scripts/alignRMSD.pl'
ALIGN_SCRIPT = '/cs/staff/dina/scripts/align.pl'
RENUMBER_SCRIPT = '/cs/staff/dina/utils/renumber'
PARAMS_SCRIPT = '/cs/staff/dina/PatchDock/buildParams.pl'
DOCK_SCRIPT = '/cs/staff/dina/PatchDock/patch_dock.Linux'
PD_TRANS_SCRIPT = '/cs/staff/dina/PatchDock/PatchDockOut2Trans.pl'
PRE_COMBDOCK_SCRIPT = '/cs/staff/dina/projects/CombDock/bin/PatchDock2CombDock.pl'
CHANGE_CHAIN_ID_SCRIPT = '/cs/staff/dina/scripts/chainger.pl'
XL_SCORE_PROG = '/cs/labs/dina/seanco/xl_score_prob/cross_link_score'
SOAP_ENV = '/cs/staff/dina/libs/imp_build/setup_environment.sh'
SOAP_RUN = '/cs/staff/dina/libs/imp_build/bin/soap_score'
SLURM_OUT_DIR = '/cs/labs/dina/seanco/xl_mlp_nn/slurm/'
#


def unit_prob_and_reg_score_files(out_dir, chains, name):
    reg_score_file = f"{out_dir}out_score_regular_{chains[0]}_{chains[1]}.txt"
    prob_score_file = f"{out_dir}out_score_prob_{chains[0]}_{chains[1]}.txt"
    new_file = f"{out_dir}all_scores_{name}.csv"
    prob_violation, prob_score, reg_violation, reg_score, soap_score, rmsd, i_rmsd, dockQ = [], [], [], [], [], [], [], []
    with open(prob_score_file, 'r') as f:
        for line in f:
            words = line.split()
            prob_violation.append(float(words[0]))
            prob_score.append(float(words[1]))
    with open(reg_score_file, 'r') as f:
        for line in f:
            words = line.split()
            reg_violation.append(float(words[0]))
            reg_score.append(float(words[1]))
            rmsd.append(float(words[2]))
            dockQ.append(float(words[3]))
            i_rmsd.append(float(words[4]))
            soap_score.append(float(words[5]))
    rows = zip(reg_violation, reg_score, prob_violation, prob_score, rmsd, i_rmsd, soap_score, dockQ)
    with open(new_file, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(("reg_violations", "reg_score", "prob_violations", "prob_score", "rmsd", "interface_rmsd", "soap_score", "dockQ"))
        wr.writerows(rows)


def change_file_names(offset=60):
    dir_path = '/cs/labs/dina/seanco/xl_mlp_nn/predictions/6nr8/'
    files = os.listdir(dir_path)
    for file in files:
        if file[:7] == 'complex' and file.split('.')[-1] == 'pdb':
            new_i = offset + int(file.split('.')[0][7:])
            new_name = f"complex{new_i}.pdb"
            os.rename(dir_path + file, dir_path + new_name)


def find_new_chain_tmp(cur_chain, chains):
    new_chain = ""
    for c in cur_chain:
        if c in chains:
            new_chain += c
    if len(new_chain) > 0:
        return 1, new_chain
    return 0, cur_chain


def create_tmp_xl_file(file_path, out_dir, chains):
    new_file = out_dir + 'tmp_xl_file.txt'
    in_f = open(file_path, 'r')
    out_f = open(new_file, 'w')
    num_xl = 0
    for line in in_f:
        words = line.split()
        tmp1, words[1] = find_new_chain_tmp(words[1], chains)
        tmp2, words[3] = find_new_chain_tmp(words[3], chains)
        new_line = " ".join(words)
        out_f.write(new_line)
        out_f.write('\n')
        if words[1] != words[3] or len(words[1]) > 1:
            num_xl += tmp1 + tmp2
    out_f.close()
    in_f.close()
    return new_file, num_xl


def write_score_file(violations, score, rmsd, out_path):
    with open(out_path, 'w') as f:
        for i in range(len(rmsd)):
            f.write(f"{violations[i]} {score[i]} {rmsd[i]}\n")


def read_xl_score_output(res_file):
    violations = []
    score = []
    with open(res_file, 'r') as f:
        f.readline()
        for line in f:
            words = line.split()
            violations.append(int(words[2]))
            score.append(float(words[3]))
    return violations, score


def read_rmsd_from_clusters(dir_path):
    clusters_file = dir_path + 'clusters.res'
    rmsd = []
    with open(clusters_file, 'r') as f:
        for line in f:
            words = line.split()
            rmsd.append(float(words[5]))
    return rmsd


def read_combdock_rmsds(solution_pdb, pdb_models, out_path):
    rmsd = []
    if solution_pdb == 'clusters':
        rmsd = read_rmsd_from_clusters(out_path)
    else:
        for file in pdb_models:
            if file.split('.')[-1] == 'pdb':
                cmd = f"{ALIGN_RMSD_SCRIPT} {solution_pdb} {file}"
                res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
                output = res.stdout
                lines = output.split()
                rmsd.append(float(lines[-1]))
    return rmsd


def sort_files_lambda(f):
    key = f.split('/')[-1].split('.')[0]
    if len(key) == 8:
        return int(key[-1])
    else:
        return int(key[-2:])


def run_xl_score(out_path, th_file, regular_xl, prob_xl, solution_pdb):
    complexes_path = f"{out_path}complexes/"
    files = os.listdir(complexes_path)
    files = [complexes_path + file for file in files]
    files = sorted(files, key=sort_files_lambda)
    out_prob = out_path + "prob_xl_score.res"
    out_reg = out_path + "regular_xl_score.res"
    reg_args = [XL_SCORE_PROG, regular_xl] + files + ['-o', out_reg]
    prob_args = [XL_SCORE_PROG, prob_xl] + files + ['-o', out_prob, '-t', th_file]
    subprocess.Popen(reg_args).communicate()
    violations_reg, score_reg = read_xl_score_output(out_reg)
    subprocess.Popen(prob_args).communicate()
    violations_prob, score_prob = read_xl_score_output(out_prob)
    rmsd = read_combdock_rmsds(solution_pdb, files, out_path)
    return violations_reg, violations_prob, score_reg, score_prob, rmsd


def compare_combdock_xl_score(out_path, th_file, regular_xl, prob_xl, name, solution_pdb):
    violations_reg, violations_prob, score_reg, score_prob, rmsd = run_xl_score(out_path, th_file,
                                                                                regular_xl, prob_xl, solution_pdb)
    out_reg, out_prob = out_path + 'regular_out.txt', out_path + 'prob_out.txt'
    write_score_file(violations_reg, score_reg, rmsd, out_reg)
    write_score_file(violations_prob, score_prob, rmsd, out_prob)
    general_utils.plot_funnel(r=2, s=1, v=0, t=f"Regular Restrains {name}", file_path=out_reg,
                              negate_score=False, color='m', save=False, prob_score_weight=None, normalize=True,
                              filter_th=100, soap=False, soap_weight=0, i_rmsd=False, plot_soap=False, dir_path=out_path)
    general_utils.plot_funnel(r=2, s=1, v=0, t=f"Probability Restrains {name}", file_path=[out_reg, out_prob],
                              negate_score=False, color='c', save=False, prob_score_weight=0.25, filter_th=100,
                              soap=False, soap_weight=0, i_rmsd=False, plot_soap=False, dir_path=out_path)


def create_su_list_file(pdbs, out_dir):
    with open(out_dir + 'SU.list', 'w') as su:
        for pdb in pdbs:
            su.write(pdb + '.pdb\n')


def combdock_preprocess(out_dir, pdbs, pd_files):
    os.chdir(out_dir)
    pd_count = 0
    pdbs = [pdb.split('/')[-1].split('.')[0] for pdb in pdbs]
    create_su_list_file(pdbs, out_dir)
    pd_files = [pd.split('/')[-1] for pd in pd_files]
    for i in range(len(pdbs) - 1):
        for j in range(i + 1, len(pdbs)):
            pdb_str = f"best_{pdbs[i]}_plus_{pdbs[j]}"
            cmd = f"{PRE_COMBDOCK_SCRIPT} {pd_files[pd_count]} > {pdb_str}"
            subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()
            pd_count += 1


def run_af_on_existing_pdb(pdb_files, af_out_dir, generate_fasta=True, fasta_dir=FASTA_PATH):
    for file in pdb_files:
        fasta_path = fasta_dir + file.split('/')[-1].split('.')[0] + '.fasta'
        if generate_fasta and not os.path.isfile(fasta_path):
            cmd = f"{PDB_2_FASTA} {file} | tee {fasta_path}"
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            seq = str(p.communicate()[0]).split()[-1]
    os.chdir(SLURM_OUT_DIR)
    cmd = collabfold_script_moriah.run_colabfold_batch(fasta_dir, '', run_script=False, out_dir=af_out_dir)
    p = subprocess.Popen("sbatch -W " + cmd, shell=True, stdout=subprocess.PIPE)
    p.communicate()
    to_remove = []
    copied = 0
    new_file_names = []
    for pdb in pdb_files:
        file_chain = pdb.split('/')[-1].split('.')[0].split('_')[-1]
        for file in listdir(af_out_dir):
            suff = file.split('.')[-1]
            words = file.split('_')
            if len(words) > 5 and suff == 'pdb' and words[-3] == '1' and words[-6] == file_chain:
                dest_dir = alpha_fold_files.AF_PDB_DIR
                new_file_name = dest_dir + words[-6] + '.' + suff
                shutil.copy(af_out_dir + file, new_file_name)
                new_file_names.append(new_file_name)
                copied += 1
            if suff != '.txt':
                to_remove.append(file)

        # for file in to_remove:
        #     if not os.path.isdir(af_out_dir + file):
        #         os.remove(af_out_dir + file)
    return new_file_names


def change_chain_id_in_pdb(pdb_name):
    correct_id = pdb_name.split('/')[-1].split('.')[0]
    if correct_id != "A":
        subprocess.Popen(f"{CHANGE_CHAIN_ID_SCRIPT} {pdb_name} A {correct_id}", shell=True).communicate()


def align_af_to_pdb(pdb_file, af_file):
    base_name = ".".join(af_file.split('.')[:-1])
    aligned_file = base_name + '_tmp.pdb'
    cmd = ALIGN_SCRIPT + " " + pdb_file + " " + af_file + " " + aligned_file
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()
    pdb_parser = PDBParser(PERMISSIVE=1)
    structure = pdb_parser.get_structure(pdb_file.split('/')[-1], pdb_file)
    chain = list(structure.get_chains())[0]
    first_res = chain.child_list[0].id[1]
    new_aligned_file = base_name + '_tr.pdb'
    if first_res > 1:
        subprocess.Popen(f"{RENUMBER_SCRIPT} {aligned_file} {first_res - 1} > {new_aligned_file}", shell=True).communicate()
    else:
        shutil.copy(aligned_file, new_aligned_file)
    return new_aligned_file


def convert_chains_to_af_structure(pdbs, af_out_dir='/cs/labs/dina/seanco/xl_mlp_nn/af_align/', skip_exist=False, fasta_dir=FASTA_PATH):
    aligned = [af_out_dir + pdb.split('/')[-1].split('.')[0].split('_')[-1] + "_tr.pdb" for pdb in pdbs]
    is_exist = [os.path.isfile(al) for al in aligned]
    if skip_exist and (False not in is_exist):
        return aligned
    af_files = run_af_on_existing_pdb(pdbs, af_out_dir, fasta_dir=fasta_dir)
    aligned = []
    for i, af_file in enumerate(af_files):
        change_chain_id_in_pdb(af_file)
        aligned.append(align_af_to_pdb(pdbs[i], af_file))
    return aligned


def run_pd_parallel(aligned_files, out_dir):
    os.chdir(out_dir)
    processes = []
    for i in range(len(aligned_files) - 1):
        for j in range(i + 1, len(aligned_files)):
            pd_res_file = f"{out_dir}{aligned_files[i].split('/')[-1].split('_')[0]}_{aligned_files[j].split('/')[-1].split('_')[0]}_pd_res.txt"
            if not os.path.isfile(pd_res_file):
                cmd_params = PARAMS_SCRIPT + " " + aligned_files[i] + " " + aligned_files[j]
                cmd_dock = DOCK_SCRIPT + f" params.txt {pd_res_file}"
                trans_file = f"trans_{aligned_files[i].split('/')[-1].split('_')[0]}_{aligned_files[j].split('/')[-1].split('_')[0]}.txt"
                soap_file = f"soap_score_{aligned_files[i].split('/')[-1].split('_')[0]}_{aligned_files[j].split('/')[-1].split('_')[0]}.res"
                trand_cmd = f"{PD_TRANS_SCRIPT} {pd_res_file} > {trans_file}"
                soap_cmd = f"{SOAP_ENV} {SOAP_RUN} {aligned_files[i]} {aligned_files[j]} {trans_file} -o {soap_file}"
                cmd = f"{cmd_params}; {cmd_dock}; {trand_cmd}; {soap_cmd}"
                p = subprocess.Popen(cmd, shell=True)
                processes.append(p)
    for p in processes:
        p.communicate()


def dock_pdbs(aligned_af_a, aligned_af_b, out_dir='/cs/labs/dina/seanco/xl_mlp_nn/af_align/', skip_exist=False):
    os.chdir(out_dir)
    pd_res_file = f"{out_dir}{aligned_af_a.split('/')[-1].split('_')[0]}_{aligned_af_b.split('/')[-1].split('_')[0]}_pd_res.txt"
    if not (skip_exist and os.path.isfile(pd_res_file)):
        cmd = PARAMS_SCRIPT + " " + aligned_af_a + " " + aligned_af_b
        subprocess.Popen(cmd, shell=True).communicate()
        cmd = DOCK_SCRIPT + f" params.txt {pd_res_file}"
        subprocess.Popen(cmd, shell=True).communicate()
    return pd_res_file


def trans_and_run_soap(pd_out_file, aligned_a, aligned_b, skip_exists=False):
    trans_file = f"trans_{aligned_a.split('/')[-1].split('_')[0]}_{aligned_b.split('/')[-1].split('_')[0]}.txt"
    soap_file = f"soap_score_{aligned_a.split('/')[-1].split('_')[0]}_{aligned_b.split('/')[-1].split('_')[0]}.res"
    if skip_exists and os.path.isfile(soap_file):
        return
    subprocess.Popen(f"{PD_TRANS_SCRIPT} {pd_out_file} > {trans_file}", shell=True).communicate()
    cmd = f"{SOAP_ENV} {SOAP_RUN} {aligned_a} {aligned_b} {trans_file} -o {soap_file}"
    subprocess.Popen(cmd, shell=True).communicate()


def create_af_align_run_pd(pdbs, out_dir='/cs/labs/dina/seanco/xl_mlp_nn/af_align/', skip_exist=False,
                           af_out_dir='/cs/labs/dina/seanco/xl_mlp_nn/af_align/', bound_form=False):
    base_name = out_dir.split('/')[-2]
    fasta_dir = FASTA_PATH + base_name + '/'
    if not os.path.exists(fasta_dir):
        os.makedirs(fasta_dir)
    af_out_dir_path = af_out_dir + base_name + '/'
    if not os.path.exists(af_out_dir_path):
        os.makedirs(af_out_dir_path)
    if not bound_form:
        aligned_pdbs = convert_chains_to_af_structure(pdbs, af_out_dir_path, skip_exist, fasta_dir)
    else:
        aligned_pdbs = pdbs
    if aligned_pdbs is not None and len(aligned_pdbs) > 0:
        if bound_form:
            new_aligned = [out_dir + aligned.split('/')[-1].split('_')[-1].split('.')[0] + '_tr.pdb' for aligned in aligned_pdbs]
        else:
            new_aligned = [out_dir + aligned.split('/')[-1] for aligned in aligned_pdbs]
        for i in range(len(new_aligned)):
            if not (skip_exist and os.path.isfile(new_aligned[i])):
                shutil.copy(aligned_pdbs[i], new_aligned[i])
        pd_files = []
        # run_pd_parallel(new_aligned, out_dir)
        for i in range(len(new_aligned) - 1):
            for j in range(i + 1, len(new_aligned)):
                pd_file = dock_pdbs(new_aligned[i], new_aligned[j], out_dir, skip_exist)
                pd_files.append(pd_file)
                trans_and_run_soap(pd_file, new_aligned[i], new_aligned[j], skip_exist)
        return new_aligned, pd_files
    return None


def compare_xl_restraints(pdb_a, pdb_b, pd_file, regular_xl, prob_xl, th_file, out_path, name):
    a_name = pdb_a.split('/')[-1].split('_')[0]
    b_name = pdb_b.split('/')[-1].split('_')[0]
    out_name_reg = f"out_score_regular_{a_name}_{b_name}.txt"
    out_soap_path = f"{out_path}soap_score_{a_name}_{b_name}.res"
    tmp_xl_file, num_xl = create_tmp_xl_file(regular_xl, out_path, {a_name, b_name})
    if num_xl < 2:
        os.remove(tmp_xl_file)
        return
    p = subprocess.Popen([XL_DOCKING_SCORE_EXE, pdb_a, pdb_b, pd_file,
                          tmp_xl_file, 'None', out_path + out_name_reg, out_soap_path])
    p.communicate()
    os.remove(tmp_xl_file)

    out_name_prob = f"out_score_prob_{a_name}_{b_name}.txt"
    tmp_xl_file, num_xl = create_tmp_xl_file(prob_xl, out_path, {a_name, b_name})
    p = subprocess.Popen([XL_DOCKING_SCORE_EXE, pdb_a, pdb_b, pd_file,
                          tmp_xl_file, th_file, out_path + out_name_prob, out_soap_path])
    p.communicate()
    os.remove(tmp_xl_file)
    out_reg = out_path + out_name_reg
    out_prob = out_path + out_name_prob
    general_utils.plot_funnel(r=2, s=1, v=0, t=f"{name}_{a_name}_{b_name}",
                              file_path=out_path + out_name_reg, negate_score=False, color='c', save=True,
                              prob_score_weight=None,  normalize=True, filter_th=100, soap=True, soap_weight=0,
                              i_rmsd=True, plot_soap=True, dir_path=out_path)
    # general_utils.plot_funnel(r=2, s=1, v=0, t=f"Probability Restrains {name} {a_name} {b_name}",
    #                           file_path=out_path + out_name_prob, negate_score=True, color='m', save=False)
    general_utils.plot_funnel(r=2, s=1, v=0, t=f"{name}_{a_name}_{b_name}",
                              file_path=[out_reg, out_prob], negate_score=False, color='m', save=True,
                              prob_score_weight=0.2,  normalize=True, filter_th=100, soap=True, soap_weight=0.2,
                              reg_score_weight=0.6, i_rmsd=True, plot_soap=False, dir_path=out_path)
    general_utils.plot_funnel(r=2, s=1, v=0, t=f"{name}_{a_name}_{b_name}",
                              file_path=[out_reg, out_prob], negate_score=False, color='m', save=True,
                              prob_score_weight=1,  normalize=True, filter_th=100, soap=True, soap_weight=0,
                              reg_score_weight=0, i_rmsd=True, plot_soap=False, dir_path=out_path)


def create_pd_file_list(out_dir, chains):
    pd_files = []
    for i in range(len(chains) - 1):
        for j in range(i + 1, len(chains)):
            pd_res_file = f"{out_dir}{chains[i]}_{chains[j]}_pd_res.txt"
            pd_files.append(pd_res_file)
    return pd_files


def create_aligned_pdb_file_list(pdbs, out_dir):
    aligned = []
    for pdb in pdbs:
        chain = pdb.split('/')[-1].split('.')[0].split('_')[-1]
        aligned.append(out_dir + chain + '_tr.pdb')
    return aligned


def parse_args():
    parser = CompParser()
    args = parser.parse_args()
    return args


class CompParser(argparse.ArgumentParser):

    def __init__(self, **kwargs):
        super(CompParser, self).__init__(**kwargs)

        self.add_argument(
            "--name",
            help="Name of the protein",
            default='2BBM',
            type=str
        )

        self.add_argument(
            "--regular_xl",
            help="Path to regular Cross Links file in format: residue_number_a chain_id_a residue_number_b "
                 "chain_id_b min_distance max_distance",
            default='/cs/labs/dina/seanco/xl_mlp_nn/predictions/2bbm/pdb2bbm.txt',
            # default='/cs/labs/dina/seanco/xl_mlp_nn/predictions/6nr8/pdb6nr8.txt',
            type=str,
        )

        self.add_argument(
            "--prob_xl",
            help="Path to probability Cross Links file in format: residue_number_a chain_id_a residue_number_b "
                 "chain_id_b min_distance max_distance [prediction_probabilities]",
            # default='/cs/labs/dina/seanco/DockingXlScore/data/CASP13_target/fake_prediction.txt',
            default='/cs/labs/dina/seanco/xl_mlp_nn/predictions/2bbm/pred_xlinks.txt',
            type=str,
        )

        self.add_argument(
            "--solution_pdb",
            help="Path to pdb solution file",
            # default='/cs/labs/dina/seanco/DockingXlScore/data/CASP13_target/A.pdb',
            # default='clusters',
            default='/cs/labs/dina/seanco/xl_mlp_nn/predictions/2bbm/pdb2bbm.pdb',
            type=str
        )

        self.add_argument(
            "--pdbs",
            help="Path to pdb chain b file",
            # default=['/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_A.ent',
            #          '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_B.ent',
            #          '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_C.ent',
            #          '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_D.ent',
            #          '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_E.ent',
            #          '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_F.ent',
            #          '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_G.ent',
            #          '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_H.ent',
            #          '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_I.ent',
            #          '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_J.ent',
            #          '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_K.ent',
            #          '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_L.ent'],
            default=['/cs/labs/dina/seanco/xl_parser/pdbs/pdb2bbm_A.ent',
                     '/cs/labs/dina/seanco/xl_parser/pdbs/pdb2bbm_B.ent'],
                     # '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_D.ent',
                     # '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_E.ent',
                     # '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_F.ent',
                     # '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_G.ent',
                     # '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_H.ent',
                     # '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_I.ent',
                     # '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_J.ent',
                     # '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_K.ent',
                     # '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_L.ent',
                     # '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_M.ent',
                     # '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_N.ent',
                     # '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_O.ent',
                     # '/cs/labs/dina/seanco/xl_parser/pdbs/pdb6nr8_P.ent'],
            nargs='+'
        )

        self.add_argument(
            "--out_path",
            help="Path scores output file",
            default='/cs/labs/dina/seanco/xl_mlp_nn/predictions/2bbm/',
            type=str
        )
        self.add_argument(
            "--th_file",
            help="Path to distances threshold file",
            default='/cs/labs/dina/seanco/xl_mlp_nn/predictions/2bbm/th_file.txt',
            type=str
        )
        self.add_argument(
            "--pd_file",
            help="Path to PatchDock results file",
            # default='/cs/labs/dina/seanco/xl_mlp_nn/af_align/pd_res.txt',
            # default='/cs/labs/dina/seanco/DockingXlScore/data/CASP13_target/output_0.8.res',
            default='',
            type=str
        )
        self.add_argument(
            "--mode",
            help="run mode: 'comp' for comparison only, 'af_align' for af model creation and align, 'both' for both",
            default='both',
            type=str
        )

    def parse_args(self, args=None, namespace=None):
        """ Parse the input arguments """
        args = super(CompParser, self).parse_args(args, namespace)
        return args


def main():
    _args = parse_args()
    if not os.path.exists(_args.out_path):
        os.makedirs(_args.out_path)
    if _args.mode == 'af_align':
        create_af_align_run_pd(_args.pdbs, _args.out_path, skip_exist=True, bound_form=False)
    elif _args.mode == 'comp':
        compare_xl_restraints(_args.pdb_a, _args.pdb_b, _args.pd_file, _args.regular_xl, _args.prob_xl,
                              _args.th_file, _args.out_path, _args.name)
    elif _args.mode == 'pre_combdock':
        # pd_files = create_pd_file_list(_args.out_path, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        # pd_files = create_pd_file_list(_args.out_path, ['A', 'B', 'C', 'D'])
        pd_files = create_pd_file_list(_args.out_path, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'])
        aligned_pdbs = create_aligned_pdb_file_list(_args.pdbs, _args.out_path)
        combdock_preprocess(_args.out_path, aligned_pdbs, pd_files)
    elif _args.mode == 'comp_combdock':
        # change_file_names()
        compare_combdock_xl_score(_args.out_path, _args.th_file, _args.regular_xl, _args.prob_xl, _args.name,
                                  _args.solution_pdb)
    elif _args.mode == 'both':
        # create_af_align_run_pd(_args.pdbs, _args.out_path, skip_exist=False, bound_form=False)
        # change_file_names()
        # unit_prob_and_reg_score_files(_args.out_path, ['A', 'B'], _args.name)
        # unit_prob_and_reg_score_files('/cs/labs/dina/seanco/xl_mlp_nn/predictions/6nr8/', ['C', 'D'], '6nr8_C_D')
        # general_utils.plot_funnel(r=2, s=1, v=0, t='Regular Restrains 6nr8',
        #                           file_path=_args.out_path + 'out_score_regular.txt', negate_score=False, color='c', save=True)
        # aligned_pdbs, pd_files = create_af_align_run_pd(_args.pdbs, _args.out_path, skip_exist=True)
        # pd_files = create_pd_file_list(_args.out_path, ['A', 'C'])
        pd_files = create_pd_file_list(_args.out_path, ['A', 'B'])
        aligned_pdbs = create_aligned_pdb_file_list(_args.pdbs, _args.out_path)
        # Must predict between these steps
        pd_count = 0
        for i in range(len(aligned_pdbs) - 1):
            for j in range(i + 1, len(aligned_pdbs)):
                compare_xl_restraints(aligned_pdbs[i], aligned_pdbs[j], pd_files[pd_count], _args.regular_xl, _args.prob_xl,
                                      _args.th_file, _args.out_path, _args.name)
                pd_count += 1


if __name__ == "__main__":
    main()
