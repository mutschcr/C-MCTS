def print_jobscript(iParams, folder_path, tag, time, mode):
    with open(f"{folder_path}/job_{tag}.sh", 'w', newline='\n') as f:
        print("#!/bin/bash -l", file=f)
        print(f"#SBATCH --time={time}:00:00", file=f)
        print(f"#SBATCH --job-name={tag}", file=f)
        print(f"#SBATCH --job-name={tag}", file=f)
        print(f"#SBATCH --output=out_{tag}", file=f)
        print(f"#SBATCH --export=NONE", file=f)
        print(f"#SBATCH --export=NONE", file=f)
        print(f"unset SLURM_EXPORT_ENV", file=f)
        print(f"module load python", file=f)
        print(f"source ~/.bashrc", file=f)
        print(f"source activate pybind", file=f)
        print(f"cd ~/safe_mcts/SafeMCTS/src_python", file=f)
        print(
            f"srun python run.py rocksample_7_8 {mode} {iParams[0]} {iParams[4]} {iParams[2]} {iParams[3]} 1.0 {iParams[1]}",
            file=f)
