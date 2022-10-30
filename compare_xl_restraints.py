import sys
import subprocess
import argparse
import os
from os import listdir
import shutil
sys.path.insert(0, '/cs/labs/dina/seanco/xl_parser')
import general_utils
import alpha_fold_files
from alphafold import collabfold_script_moriah

XL_DOCKING_SCORE_EXE = "/cs/labs/dina/seanco/DockingXlScore/cmake-build-debug/DockingXlScore"
PDB_2_FASTA = '/cs/staff/dina/utils/pdb2fasta'
FASTA_PATH = '/cs/labs/dina/seanco/xl_parser/fasta_files/'
RUN_ALPHAFOLD_SCRIPT = '/cs/labs/dina/seanco/xl_parser/alphafold/collabfold_script_moriah.py'
ALIGN_RMSD_SCRIPT = '/cs/staff/dina/scripts/alignRMSD.pl'
ALIGN_SCRIPT = '/cs/staff/dina/scripts/align.pl'
PARAMS_SCRIPT = '/cs/staff/dina/PatchDock/buildParams.pl'
DOCK_SCRIPT = '/cs/staff/dina/PatchDock/patch_dock.Linux'
PD_TRANS_SCRIPT = '/cs/staff/dina/PatchDock/PatchDockOut2Trans.pl'
CHANGE_CHAIN_ID_SCRIPT = '/cs/staff/dina/scripts/chainger.pl'
SOAP_ENV = '/cs/staff/dina/libs/imp_build/setup_environment.sh'
SOAP_RUN = '/cs/staff/dina/libs/imp_build/bin/soap_score'
SLURM_OUT_DIR = '/cs/labs/dina/seanco/xl_mlp_nn/slurm/'


def run_af_on_existing_pdb(pdb_file, af_out_dir, generate_fasta=False):
    seq = None
    fasta_path = FASTA_PATH + pdb_file.split('/')[-1].split('.')[0] + '.fasta'
    if generate_fasta:
        cmd = f"{PDB_2_FASTA} {pdb_file} | tee {fasta_path}"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        seq = str(p.communicate()[0]).split()[-1]
    new_file_name = ""
    if seq is None or len(seq) <= 1700:
        os.chdir(SLURM_OUT_DIR)
        cmd = collabfold_script_moriah.run_colabfold_batch(fasta_path, '', run_script=False, out_dir=af_out_dir)
        p = subprocess.Popen("sbatch -W " + cmd, shell=True, stdout=subprocess.PIPE)
        p.communicate()
        to_remove = []
        copied = 0
        file_chain = pdb_file.split('/')[-1].split('.')[0]
        for file in listdir(af_out_dir):
            suff = file.split('.')[-1]
            words = file.split('_')
            if len(words) > 5 and suff == 'pdb' and words[-3] == '1' and words[-6] == file_chain:
                dest_dir = alpha_fold_files.AF_PDB_DIR
                new_file_name = dest_dir + words[-6] + '.' + suff
                shutil.copy(af_out_dir + file, new_file_name)
                copied += 1
            if suff != '.txt':
                to_remove.append(file)

        # for file in to_remove:
        #     if not os.path.isdir(af_out_dir + file):
        #         os.remove(af_out_dir + file)
    else:
        print(f"Too large protein. length: {len(seq)}")
    return new_file_name


def change_chain_id_in_pdb(pdb_name):
    correct_id = pdb_name.split('/')[-1].split('.')[0]
    if correct_id != "A":
        subprocess.Popen(f"{CHANGE_CHAIN_ID_SCRIPT} {pdb_name} A {correct_id}", shell=True).communicate()


def align_af_to_pdb(pdb_file, af_file):
    aligned_file = ".".join(af_file.split('.')[:-1]) + '_tr.pdb'
    cmd = ALIGN_SCRIPT + " " + pdb_file + " " + af_file + " " + aligned_file
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()
    return aligned_file


def convert_chains_to_af_structure(pdb_a, pdb_b, out_dir='/cs/labs/dina/seanco/xl_mlp_nn/af_align/', skip_exist=False):
    aligned_a = alpha_fold_files.AF_PDB_DIR + pdb_a.split('/')[-1].split('.')[0] + "_tr.pdb"
    aligned_b = alpha_fold_files.AF_PDB_DIR + pdb_b.split('/')[-1].split('.')[0] + "_tr.pdb"
    if skip_exist and os.path.isfile(aligned_a) and os.path.isfile(aligned_b):
        return aligned_a, aligned_b
    af_file = run_af_on_existing_pdb(pdb_a, out_dir)
    if af_file != "":
        change_chain_id_in_pdb(af_file)
        aligned_a = align_af_to_pdb(pdb_a, af_file)
        af_file = run_af_on_existing_pdb(pdb_b, out_dir)
        if af_file != "":
            change_chain_id_in_pdb(af_file)
            aligned_b = align_af_to_pdb(pdb_b, af_file)
            return aligned_a, aligned_b
    return None, None


def dock_pdbs(aligned_af_a, aligned_af_b, out_dir='/cs/labs/dina/seanco/xl_mlp_nn/af_align/', skip_exist=False):
    os.chdir(out_dir)
    pd_res_file = out_dir + 'pd_res.txt'
    if not (skip_exist and os.path.isfile(pd_res_file)):
        cmd = PARAMS_SCRIPT + " " + aligned_af_a + " " + aligned_af_b
        subprocess.Popen(cmd, shell=True).communicate()
        cmd = DOCK_SCRIPT + f" params.txt {pd_res_file}"
        subprocess.Popen(cmd, shell=True).communicate()
    return pd_res_file


def trans_and_run_soap(pd_out_file, aligned_a, aligned_b, skip_exists=False):
    if skip_exists and os.path.isfile("soap_score.res"):
        return
    subprocess.Popen(f"{PD_TRANS_SCRIPT} {pd_out_file} > trans", shell=True).communicate()
    cmd = f"{SOAP_ENV} {SOAP_RUN} {aligned_a} {aligned_b} trans"
    subprocess.Popen(cmd, shell=True).communicate()


def create_af_align_run_pd(pdb_a, pdb_b, out_dir='/cs/labs/dina/seanco/xl_mlp_nn/af_align/', skip_exist=False):
    aligned_a, aligned_b = convert_chains_to_af_structure(pdb_a, pdb_b, out_dir, skip_exist)
    pd_file = None
    if aligned_a is not None:
        pd_file = dock_pdbs(aligned_a, aligned_b, out_dir, skip_exist)
        trans_and_run_soap(pd_file, aligned_a, aligned_b, skip_exist)
    return aligned_a, aligned_b, pd_file


def compare_xl_restraints(pdb_a, pdb_b, pd_file, regular_xl, prob_xl, th_file, out_path, name):
    p = subprocess.Popen([XL_DOCKING_SCORE_EXE, pdb_a, pdb_b, pd_file,
                          regular_xl, 'None', out_path + 'out_score_regular.txt'])
    p.communicate()
    p = subprocess.Popen([XL_DOCKING_SCORE_EXE, pdb_a, pdb_b, pd_file,
                          prob_xl, th_file, out_path + 'out_score_prob.txt'])
    p.communicate()
    general_utils.plot_funnel(r=2, s=1, v=0, t='Regular Restrains ' + name,
                              file_path=out_path + 'out_score_regular.txt', negate_score=False, color='c')
    general_utils.plot_funnel(r=2, s=1, v=0, t='Probability Restrains ' + name,
                              file_path=out_path + 'out_score_prob.txt', negate_score=True, color='m')


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
            default='Casp13_target',
            type=str
        )

        self.add_argument(
            "--regular_xl",
            help="Path to regular Cross Links file in format: residue_number_a chain_id_a residue_number_b "
                 "chain_id_b min_distance max_distance",
            default='/cs/labs/dina/seanco/DockingXlScore/data/CASP13_target/xlinks.txt',
            type=str,
        )

        self.add_argument(
            "--prob_xl",
            help="Path to probability Cross Links file in format: residue_number_a chain_id_a residue_number_b "
                 "chain_id_b min_distance max_distance [prediction_probabilities]",
            default='/cs/labs/dina/seanco/DockingXlScore/data/CASP13_target/fake_prediction.txt',
            type=str,
        )

        self.add_argument(
            "--pdb_a",
            help="Path to pdb chain a file",
            default='/cs/labs/dina/seanco/DockingXlScore/data/CASP13_target/A.pdb',
            type=str
        )

        self.add_argument(
            "--pdb_b",
            help="Path to pdb chain b file",
            default='/cs/labs/dina/seanco/DockingXlScore/data/CASP13_target/B.pdb',
            type=str
        )

        self.add_argument(
            "--out_path",
            help="Path scores output file",
            default='/cs/labs/dina/seanco/DockingXlScore/data/CASP13_target/',
            type=str
        )
        self.add_argument(
            "--th_file",
            help="Path to distances threshold file",
            default='/cs/labs/dina/seanco/DockingXlScore/data/CASP13_target/th_file.txt',
            type=str
        )
        self.add_argument(
            "--pd_file",
            help="Path to PatchDock results file",
            default='/cs/labs/dina/seanco/DockingXlScore/data/CASP13_target/output_0.8.res',
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
    if _args.mode == 'af_align':
        create_af_align_run_pd(_args.pdb_a, _args.pdb_b)
    elif _args.mode == 'comp':
        compare_xl_restraints(_args.pdb_a, _args.pdb_b, _args.pd_file, _args.regular_xl, _args.prob_xl,
                              _args.th_file, _args.out_path, _args.name)
    elif _args.mode == 'both':
        aligned_pdb_a, aligned_pdb_b, pd_file = create_af_align_run_pd(_args.pdb_a, _args.pdb_b, skip_exist=True)
        compare_xl_restraints(aligned_pdb_a, aligned_pdb_b, pd_file, _args.regular_xl, _args.prob_xl,
                              _args.th_file, _args.out_path, _args.name)


if __name__ == "__main__":
    main()
