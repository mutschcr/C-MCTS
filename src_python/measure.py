from itertools import product
import os
from render.printer import print_jobscript

import subprocess

if __name__ == "__main__":
    mode = "all"  # learn, evaluate


    def get_tag(iParams):
        return f"itr_{iParams[0]}_nloops_{iParams[1]}_alpha_{iParams[2]}_eps_{iParams[3]}_sigmamax_{iParams[4]}"


    def run(iParams):
        # create sub folder
        tag = get_tag(iParams)
        sub_folder_path = f"{folder_name}/{tag}"
        os.makedirs(sub_folder_path, exist_ok=True)
        iParams = list(iParams)
        if mode == "evaluate":
            def isValid():
                directories = [d for d in os.listdir(sub_folder_path) if
                               os.path.isdir(os.path.join(sub_folder_path, d)) and not d.startswith("eval")]
                prefix = -1
                suffix = []
                for directory in directories:
                    temp1 = int(directory.split("_")[0])
                    temp2 = float((directory.split("_")[1]))
                    if temp1 > prefix:
                        suffix.clear()
                        prefix = temp1
                        suffix.append(temp2)
                    elif temp1 == prefix:
                        suffix.append(temp2)

                if len(directories) == 0:  # critic has not been trained
                    return False
                elif len(suffix) > 1:  # critic trained is unsafe
                    return False
                else:
                    return True

            # check if the configuration has a fully trained and reliable critic for evaluation
            if not isValid():
                return

            for itr in [128, 256, 512, 1024, 2048, 4096, 8192]:
                if os.path.exists(f"{sub_folder_path}/eval_{itr}/df_results.pickle"):
                    continue
                iParams[0] = itr
                new_tag = tag + f"_eval{itr}"

                # inputs: mode, env, mcts_itr, n_loops, alpha, epsilon, sigma_max
                # 1. print batch script
                t = int((iParams[0] / 128) ** 0.5) * 2

                print_jobscript(iParams, sub_folder_path, new_tag, time=t, mode=mode)

                # 2. Call script
                subprocess.check_call(["sbatch", f"job_{new_tag}.sh"], cwd=sub_folder_path)
        else:
            # inputs: mode, env, mcts_itr, n_loops, alpha, epsilon, sigma_max
            # 1. print batch script
            t = 6 * int(iParams[0] / 256)

            print_jobscript(iParams, sub_folder_path, tag, time=t, mode=mode)

            # 2. Call script
            subprocess.check_call(["sbatch", f"job_{tag}.sh"], cwd=sub_folder_path)

    # start inputs
    set_mcts_itr = [256, 512, 1024]
    set_n_loops = [8]
    set_alpha = [1, 2, 4, 8, 12]
    set_epsilon = [0.1, 0.3]
    set_sigma_max = [0.2, 0.5]
    # end of inputs

    # create a folder for saving results
    prefix = "generated_"
    suffix = 1
    folder_name = None
    loop_condition = False
    if folder_name is None:
        loop_condition = True
    while loop_condition:
        folder_name = f"../{prefix}{suffix}"
        if not os.path.exists(f"{folder_name}"):
            os.makedirs(f"{folder_name}")
            break
        suffix += 1

    # print configuration space
    if mode == "all":
        with open('measure.py', 'r') as input_file, open(f'{folder_name}/config_space.py', 'w') as output_file:
            print_function = False
            for line in input_file:
                if "inputs" in line and "start" in line:
                    print_function = True
                if print_function:
                    output_file.write(line)
                    if "inputs" in line and "end" in line:
                        print_function = False
                        break

        # Print commit hash file
        with open(f"{folder_name}/commit_hash.txt", 'w', newline='\n') as f:
            print(subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout, file=f)

    # list of configurations
    configurations = list(product(set_mcts_itr, set_n_loops, set_alpha, set_epsilon, set_sigma_max))
    for c in configurations:
        run(c)
