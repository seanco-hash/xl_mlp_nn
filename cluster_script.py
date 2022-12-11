
import subprocess


WORKDIR = "/cs/labs/dina/seanco/xl_mlp_nn/"

out_dir = WORKDIR + 'scripts/'

ask_time = "1-0"
mem = "128G"
cpus = "6"
gpus = "1"
gpu_mem = "24g"
INTRO = "#!/bin/tcsh \n" \
        "#SBATCH --mem=" + mem + "\n" \
                                 "#SBATCH -c" + cpus + " \n" \
                                                       "#SBATCH --time=" + ask_time + "\n" \
                                                                                      "cd /cs/labs/dina/seanco/xl_mlp_nn/\n"
CPUS_ONLY = "68"
PYTHON = "python3 "
TRAIN = 'train.py'
COMPARE_XL = 'compare_xl_restraints.py'
PREDICT = 'predict.py'
TRAIN2 = 'train2.py'
SWEEPS = '/cs/labs/dina/seanco/xl_parser/xl_db_parser_venv/bin/wandb agent seanco/xl_gnn/k4v2onan'

DATASET_RUN = 'graph_dataset.py'
#
# INLINE_INTRO = "--mem=" + mem + " -c" + cpus + " --time=" + ask_time + " --gres=gpu:" + gpus + "," \
#                "vmem:" + gpu_mem + " --killable "

INLINE_INTRO = "--mem=" + mem + " -c" + cpus + " --time=" + ask_time + " --gres gpu:a100-3-40 "
# INLINE_INTRO = "--mem=" + mem + " -c" + cpus + " --time=" + ask_time + " --partition=puffin --gres=gpu:a30 "
# INLINE_INTRO = "--mem=" + mem + " -c" + cpus + " --time=" + ask_time + " --gres=gpu:a100-1-10 "
# salloc --mem=16G -c 1 --time=1:00:00 --gres=gpu:a100-1-10
# salloc --mem=8G -c 2 --time=2:00:00 -w glacier-36

INLINE_INTRO_CPU = "--mem=" + mem + " -c" + CPUS_ONLY + " --time=" + ask_time + " --exclude=sm-[01-04,07-08],sm-[17-20],sm-16,sm-15,ohm-55,ohm-61,ohm-62,ohm-64,ohm-57,ohm-58,ohm-59,ohm-60,cb-06,cb-05,cb-11,ohm-63,cb-16,cb-08,cb-13,cb-14,cb-17 "


def run_compare_xl():
    with open(WORKDIR + 'scripts/my_comp_xl.sh', 'w') as script:
        script.write("#!/bin/tcsh\n")
        script.write("cd /cs/labs/dina/seanco/xl_mlp_nn/\n")
        script.write("source /cs/labs/dina/seanco/needle/project/venv_needle/bin/activate.csh\n")
        script.write(PYTHON + WORKDIR + COMPARE_XL)
        script.write("\n")
        script.close()
    subprocess.run("sbatch " + INLINE_INTRO_CPU + WORKDIR + "scripts/my_comp_xl.sh", shell=True)


def run_combdock(out_path, p=100):
    with open(WORKDIR + 'scripts/my_combdock.sh', 'w') as script:
        script.write("#!/bin/tcsh\n")
        script.write(f"cd {out_path}\n")
        script.write('setenv LD_LIBRARY_PATH /cs/staff/dina/libs/leda/\n')
        script.write(f"/cs/staff/dina/projects/CombDock/CombDock.Linux SU.list best_ {p} {p} xl_combdock.txt -r 0.7\n")
        # script.write(f"/cs/staff/dina/projects/CombDock/CombDock.Linux SU.list best_ {p} {p} dummy_dist\n")
        script.write('/cs/staff/dina/projects/CombDock/bin/prepareComplex.pl SU.list clusters.res 1 50')
        script.write("\n")
        script.close()
    subprocess.run("sbatch " + INLINE_INTRO_CPU + WORKDIR + "scripts/my_combdock.sh", shell=True)


def train_script():
    with open(WORKDIR + 'scripts/train.sh', 'w') as cur_script:
        cur_script.write(INTRO)
        cur_script.write("source /cs/labs/dina/seanco/xl_parser/xl_db_parser_venv/bin/activate.csh\n")
        cur_script.write(PYTHON + WORKDIR + TRAIN)
        cur_script.write("\n")
        cur_script.close()
    subprocess.run("sbatch " + WORKDIR + "scripts/train.sh", shell=True)


def train_gpu_script():
    with open(WORKDIR + 'scripts/train.sh', 'w') as script:
        script.write("#!/bin/tcsh\n")
        script.write("cd /cs/labs/dina/seanco/xl_mlp_nn/\n")
        script.write("source /cs/labs/dina/seanco/xl_parser/xl_db_parser_venv/bin/activate.csh\n")
        script.write("module load cuda/11.3\n")
        script.write("module load cudnn/8.2.1\n")
        # script.write("module load cudnn/7.6.2\n")
        script.write(PYTHON + WORKDIR + TRAIN)
        script.write("\n")
        script.close()
    subprocess.run("sbatch " + INLINE_INTRO + WORKDIR + "scripts/train.sh", shell=True)


def run_prediction():
    with open(WORKDIR + 'scripts/predict.sh', 'w') as script:
        script.write("#!/bin/tcsh\n")
        script.write("cd /cs/labs/dina/seanco/xl_mlp_nn/\n")
        script.write("source /cs/labs/dina/seanco/xl_parser/xl_db_parser_venv/bin/activate.csh\n")
        script.write("module load cuda/11.3\n")
        script.write("module load cudnn/8.2.1\n")
        # script.write("module load cudnn/7.6.2\n")
        script.write(PYTHON + WORKDIR + PREDICT)
        script.write("\n")
        script.close()
    subprocess.run("sbatch " + INLINE_INTRO + WORKDIR + "scripts/predict.sh", shell=True)


def run_sweeps_script():
    with open(WORKDIR + 'scripts/sweeps.sh', 'w') as script:
        script.write("#!/bin/tcsh\n")
        script.write("cd /cs/labs/dina/seanco/xl_mlp_nn/\n")
        script.write("source /cs/labs/dina/seanco/xl_parser/xl_db_parser_venv/bin/activate.csh\n")
        script.write("module load cuda/11.3\n")
        script.write("module load cudnn/8.2.1\n")
        # script.write("module load cudnn/7.6.2\n")
        script.write(SWEEPS)
        script.write("\n")
        script.close()
    subprocess.run("sbatch " + INLINE_INTRO + WORKDIR + "scripts/train.sh", shell=True)


def graph_dataset_script():
    with open(WORKDIR + 'scripts/dataset.sh', 'w') as script:
        script.write("#!/bin/tcsh\n")
        script.write("cd /cs/labs/dina/seanco/xl_mlp_nn/\n")
        script.write("source /cs/labs/dina/seanco/xl_parser/xl_db_parser_venv/bin/activate.csh\n")
        script.write("module load cuda/11.3\n")
        script.write("module load cudnn/8.2.1\n")
        script.write(PYTHON + WORKDIR + DATASET_RUN)
        script.write("\n")
        script.close()
    subprocess.run("sbatch " + INLINE_INTRO_CPU + WORKDIR + "scripts/dataset.sh", shell=True)


def main():
    train_gpu_script()
    # run_combdock(out_path='/cs/labs/dina/seanco/xl_mlp_nn/predictions/6nr8/', p=1000)
    # run_compare_xl()
    # run_prediction()
    # run_sweeps_script()


if __name__ == "__main__":
    main()
