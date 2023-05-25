import multiprocessing
import os
import sys
import shutil
from environment.rocksample import Rocksample
from environment.safegridworld import Safegridworld
from planner.MCTS import MCTS
import planner.MCTS as mcts_mod
import time
import copy
import pickle
import pandas as pd
import numpy.random as rand
import numpy as np
import torch
from NeuralNetworks.trainer import Trainer
import NeuralNetworks.trainer as trainer_mod
from NeuralNetworks.SafetyCritic import Ensemble

global n_sim, num_rocks, size, environment
n_sim = 8
num_rocks = 1
size = 1
if __name__ == "__main__":
    multiprocessing.set_start_method('fork')


    class SafeMCTS:
        def __init__(self, num_obs, num_actions, learning_rate, export_path, cost_constraint):
            self.total_timesteps = 1000
            self.trainer = Trainer(num_obs=num_obs, num_actions=num_actions, lr=learning_rate, folder_save=export_path)
            self.lr = learning_rate
            self.num_obs = num_obs
            self.num_actions = num_actions
            self._lambda = 0.6
            self._cost_constraint = cost_constraint
            self.export_path = export_path
            self._render = False
            self.render_path = export_path
            self.verbose = 0
            self.n_ensemble = 5
            self.lamdaMax = 30

        @property
        def render(self):
            return self._render

        @render.setter
        def render(self, value):
            self._render = value

        @property
        def lagrange_mul(self):
            return self._lambda

        @lagrange_mul.setter
        def lagrange_mul(self, value):
            self._lambda = value

        @property
        def cost_constraint(self):
            return self._cost_constraint

        @cost_constraint.setter
        def cost_constraint(self, value):
            self._cost_constraint = value

        def run_episodes(self, process_id, t_timesteps, models, use_safety_critic, save_transitions=False, isreal=True):
            s_all_list = []
            a_all_list = []
            Qc_all_list = []
            Qc_NN_list = []
            n_timesteps = 0
            V_list = []
            Vc_list = []
            R_list = []
            C_list = []
            num_steps_list = []
            n = 0
            n_pruned_list = []
            tree_depth_list = []
            global num_rocks, size
            while (mode == 'all' and n_timesteps < t_timesteps) or (mode == 'evaluate' and n < 100 / multiprocessing.cpu_count()):
                global n_sim, environment
                if "rocksample" in environment:
                    envs = [Rocksample(size, num_rocks, 0.95, True) for _ in range(multiprocessing.cpu_count())]
                    env = envs[process_id]
                elif "safegridworld" in environment:
                    if isreal:
                        envs = [Safegridworld(8, 7, 0.95, "real") for _ in range(multiprocessing.cpu_count())]
                        env = envs[process_id]
                    else:
                        envs = [Safegridworld(8, 7, 0.95, "hifi") for _ in range(multiprocessing.cpu_count())]
                        env = envs[process_id]
                if use_safety_critic:
                    safety_critic_NN = Ensemble(list_models=models, n_out=self.num_actions)
                else:
                    safety_critic_NN = None
                if "rocksample" in environment:
                    mcts_planner = MCTS(Rocksample(size, num_rocks, 0.95, False), self.num_actions, n_sim, 1, 132,
                                        0.95,
                                        mcts_mod.c_hat, safety_critic=safety_critic_NN)
                elif "safegridworld" in environment:
                    mcts_planner = MCTS(Safegridworld(8, 7, 0.95, "lofi"), self.num_actions, n_sim, 1, 132, 0.95,
                                        mcts_mod.c_hat, safety_critic=safety_critic_NN)
                start = time.time()
                done = False
                discount = 1.0
                V = 0
                Vc = 0
                R = 0
                C = 0
                i = 0
                if "safegridworld" in environment:
                    start_pos = rand.randint(3, size=2)
                    start_state = np.array([start_pos[0], start_pos[1], 0])
                    env.set_state(start_state)
                s = env.get_state()
                s_list = []
                a_list = []
                c_list = []
                Qc_list = []
                while not done and i < 100:
                    policy, root, n_pruned, tree_depth = mcts_planner.get_best_action(copy.deepcopy(s),
                                                                                      ilambda=self.lagrange_mul)
                    if use_safety_critic:
                        n_pruned_list.append(n_pruned)
                        tree_depth_list.append(tree_depth)
                    # sample actions from a policy of two actions
                    p = rand.uniform(0, 1)
                    if p < policy[0]['p']:
                        a = policy[0]['action']
                        p_taken = policy[0]['p']
                        other_cost = policy[1]['cost']
                    else:
                        a = policy[1]['action']
                        p_taken = policy[1]['p']
                        other_cost = policy[0]['cost']

                    # get cost estimated by NN at current state.
                    if use_safety_critic:
                        Qc_NN_list.append(mcts_planner.get_cost_estimate(s, a))

                    # take a step in the env, goto next state
                    rc = env.step(a)
                    c_list.append(rc[1])
                    s_list.append(s)
                    a_list.append(a)

                    # the next state
                    s = env.get_state()

                    # calculate stats (Rewards, costs etc..)
                    done = s[-1]
                    V += rc[0] * discount
                    R += rc[0]
                    Vc += rc[1] * discount
                    C += rc[1]
                    i += 1
                    discount *= 0.95

                    # reset constraint for the next step
                    mcts_planner.set_cost_constraint(rc[1], p_taken, other_cost)

                n_timesteps += i
                n += 1
                if use_safety_critic:
                    Qc_next = 0
                    discount = 1
                    for i, _ in enumerate(c_list):
                        Qc_list.append(c_list[-i - 1] + Qc_next * discount)
                        discount *= 0.95
                        Qc_next = Qc_list[-1]
                    Qc_list.reverse()
                    s_all_list += s_list
                    a_all_list += a_list
                    Qc_all_list += c_list

                if use_safety_critic or save_transitions:
                    s_next_list = s_list[1:]
                    a_next_list = a_list[1:]
                    s_list.pop()
                    a_list.pop()
                    c_list.pop()
                    self.trainer.add_transition(s_list, a_list, c_list, s_next_list, a_next_list)

                end = time.time()
                if self.verbose > 0:
                    time_per_step = (end - start) / i
                    print("")
                    print(f"EPISODE {str(n)}: ")
                    print("-----------------------------")
                    print(f"Time per step: {time_per_step :.2} s")
                    print(f"timesteps: {str(i)}")
                    print(f"discounted rewards: {str(V)}")
                    print(f"undiscounted rewards: {str(R)}")
                    print(f"discounted costs: {str(Vc)}")
                    print(f"undiscounted costs: {str(C)}")
                    print("")

                V_list.append(V)
                Vc_list.append(Vc)
                R_list.append(R)
                C_list.append(C)
                num_steps_list.append(i)
            return num_steps_list, V_list, R_list, Vc_list, C_list, s_all_list, a_all_list, Qc_all_list, Qc_NN_list, self.trainer.experience_samples.transitions, tree_depth_list

        def run_episodes_wrapper(self, args):
            return self.run_episodes(*args)

        def do_planning(self, use_safety_critic, results_save, save_transitions=False, isreal=True):
            def get_posDict(s, action, rc):
                global num_rocks, environment
                cur_position = {}
                if "rocksample" in environment:
                    num_attributes = 9
                    cur_position["action"] = action
                    cur_position["reward"] = rc[0]
                    cur_position["cost"] = rc[1]
                    cur_position["agent_x"] = s[0]
                    cur_position["agent_y"] = s[1]
                    for i in range(num_rocks):
                        cur_position[f"rock{i + 1}_x"] = s[num_attributes * i + 2]
                        cur_position[f"rock{i + 1}_y"] = s[num_attributes * i + 3]
                        cur_position[f"rock{i + 1}_val"] = s[num_attributes * i + 8]
                        cur_position[f"rock{i + 1}_collected"] = s[num_attributes * i + 9]
                elif "safegridworld" in environment:
                    cur_position["agent_x"] = s[0]
                    cur_position["agent_y"] = s[1]
                return cur_position

            os.makedirs(results_save, exist_ok=True)
            models = []
            if use_safety_critic:
                for ep in range(self.n_ensemble):
                    self.trainer.restore_checkpoint(tag=ep, cuda=torch.cuda.is_available())
                    models.append(copy.deepcopy(self.trainer.dqn))

            n_processes = multiprocessing.cpu_count()
            t_timesteps = self.total_timesteps / n_processes
            n_steps_list_all = []
            V_list_all = []
            R_list_all = []
            Vc_list_all = []
            C_list_all = []
            s_all = []
            a_all = []
            Qc_all = []
            Qc_NN_all = []
            tree_depth_all = []
            pool = multiprocessing.Pool(processes=n_processes)
            args_list = [(i, t_timesteps, models, use_safety_critic, save_transitions, isreal) for i in
                         range(n_processes)]
            results = pool.map(self.run_episodes_wrapper, args_list)
            for result in results:
                n_steps, V_list, R_list, Vc_list, C_list, s, a, Qc, Qc_NN, transitions, tree_depth = result
                n_steps_list_all += n_steps
                V_list_all += V_list
                R_list_all += R_list
                Vc_list_all += Vc_list
                C_list_all += C_list
                s_all += s
                a_all += a
                Qc_all += Qc
                Qc_NN_all += Qc_NN
                tree_depth_all += tree_depth
                self.trainer.experience_samples.transitions += transitions

            df = pd.DataFrame(list(zip(n_steps_list_all, V_list_all, R_list_all, Vc_list_all, C_list_all)),
                              columns=['time_steps', 'discounted reward', 'undiscounted reward', 'discounted cost',
                                       'undiscounted cost'])

            df.to_pickle(results_save + "df_results.pickle",
                         protocol=pickle.HIGHEST_PROTOCOL)  # Store data frame data (serialize)

            df_debug = pd.DataFrame(tree_depth_all,
                                    columns=['tree_depth'])

            df_debug.to_pickle(results_save + "df_mcts.pickle",
                               protocol=pickle.HIGHEST_PROTOCOL)  # Store data frame data (serialize)

            with open(results_save + 'stats.txt', 'w') as f:
                print(df, file=f)
                print("", file=f)
                print(f"Mean over  episodes:", file=f)
                print("------------------------------", file=f)
                print(df.mean(), file=f)

            # store test loss
            if use_safety_critic:
                df_test_loss = pd.DataFrame(list(zip(s_all, a_all, Qc_all, Qc_NN_all)),
                                            columns=["state", "action", "actual_cost", "predicted_cost"])
                df_test_loss.to_pickle(results_save + "test_loss.pickle", protocol=pickle.HIGHEST_PROTOCOL)

            # save sample transitions
            if save_transitions:
                self.trainer.store_transitions(results_save)
                self.trainer.clear()

            return df["discounted cost"].mean()

        def train_safety_critic(self, save_tag):
            results_save = f"{self.export_path}"
            self.trainer.load_transitions(results_save, train_size=0.75)
            start = time.time()
            res = self.trainer.train(epochs=200, batch_size=2048, early_stopping_patience=-1,
                                     cuda=torch.cuda.is_available())
            end = time.time()
            print(f"time to train NN per update steps :{32 * (end - start) / (1000 * 500)}")
            self.trainer.clear()

            # save results
            with open(results_save + f"train_val_loss_{save_tag}.pickle", "wb") as handle:
                pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

        def update_lagrange_mul(self, Vc, step_size):
            if Vc - self.cost_constraint < 0:
                gradient = -1
            else:
                gradient = 1
            self.lagrange_mul = np.clip(self.lagrange_mul + step_size * gradient, 0, self.lamdaMax)

        def collect_samples(self, alpha_0, eps, use_safety_critic, n_itr):
            epsilon = 1e10
            self.lagrange_mul = 0.0
            self.trainer.n_itr = n_itr - 1
            i = 0.0
            while epsilon > eps or epsilon < 0:
                print(f"LAGRANGE_MULTIPLIER={self.lagrange_mul}")
                print(f"----------------------------------")
                avg_discounted_cost = self.do_planning(use_safety_critic=use_safety_critic,
                                                       results_save=f'{self.export_path}{str(n_itr)}_{str(self.lagrange_mul)}/',
                                                       save_transitions=True, isreal=False)
                epsilon = self.cost_constraint - avg_discounted_cost
                alpha = (1.0 / (1.0 + i)) * alpha_0
                if eps >= epsilon >= 0 or (epsilon >= 0 and self.lagrange_mul == 0.0) or (
                        self.lagrange_mul == self.lamdaMax):
                    return self.lagrange_mul
                self.update_lagrange_mul(Vc=avg_discounted_cost, step_size=alpha)
                i += 1

        def learn(self, a_0, eps, n_max):

            def get_init_params():
                def get_prev_idx_alpha():
                    prev_prefix = prefix - 1
                    current_folders = [os.path.join(generated_folder, x) for x in directories if x.startswith(str(prefix)) and
                                       os.path.isdir(os.path.join(generated_folder, x))]
                    for cur_folder in current_folders:  # remove as this is going to be re-generated.
                        shutil.rmtree(cur_folder)

                    prev_folders = [x for x in directories if x.startswith(str(prev_prefix)) and
                                    os.path.isdir(os.path.join(generated_folder, x))]
                    prev_folders.sort(key=lambda f: os.path.getctime(os.path.join(generated_folder, f)),
                                      reverse=True)  # Sort the list of folders by creation time (newest first)
                    return float(prev_folders[0].split("_")[1])

                directories = [d for d in os.listdir(generated_folder) if
                               os.path.isdir(os.path.join(generated_folder, d)) and not d.startswith("eval")]
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

                if len(directories) == 0:  # generated folder is empty i.e. first run
                    return a_0, 0
                else:  # generated folder is non-empty, this is a re-run
                    if prefix == 0:
                        return a_0, 0
                    elif prefix == n_max - 1:
                        folders = [x for x in directories if x.startswith(str(prefix))]
                        folders.sort(key=lambda f: os.path.getctime(os.path.join(generated_folder, f)),
                                     reverse=True)  # Sort the list of folders by creation time (newest first)

                        if os.path.exists(
                                f"{generated_folder}{folders[0]}/exp_samples.pickle"):  # n_max counter has been reached
                            return a_0, n_max
                        else:  # job terminated before completing n_max loop
                            return get_prev_idx_alpha(), prefix

                    elif prefix > 0 and len(suffix) == 1 and suffix[0] == 0.0 and os.path.exists(
                            f"{generated_folder}{prefix}_{suffix[0]}/exp_samples.pickle"):  # training has been already completed
                        return a_0, n_max
                    else:  # training stuck in-between, needs restart
                        return get_prev_idx_alpha(), prefix

            use_safety_critic = False
            alpha_0, i = a_0, 0
            while alpha_0 > 0 and i < n_max:
                alpha_0 = self.collect_samples(alpha_0=alpha_0, eps=eps, use_safety_critic=use_safety_critic, n_itr=i)
                if alpha_0 == 0:
                    break
                for j in range(self.n_ensemble):  # train an ensemble of networks to predict safety critic.
                    self.trainer.n_itr = i
                    trainer_mod.ckp_tag = j
                    self.train_safety_critic(save_tag=f'{i}_{j}')
                use_safety_critic = True
                i += 1


    def get_tag(iParams):
        return f"itr_{iParams[0]}_nloops_{iParams[1]}_alpha_{iParams[2]}_eps_{iParams[3]}_sigmamax_{iParams[4]}"


    ablation_study = False
    env = sys.argv[1]
    mode = sys.argv[2]
    n_sim = int(sys.argv[3])  # n_simulations per time step
    stddev = float(sys.argv[4])
    alpha_0 = sys.argv[5]
    eps = sys.argv[6]
    mcts_mod.c_hat = float(sys.argv[7])
    n_max = int(sys.argv[8])
    environment = env

    folder_name = get_tag([n_sim, n_max, alpha_0, eps, stddev])
    # create a folder for saving results
    generated_folder = f"../{env}/{folder_name}/"
    if mode == "train":
        os.makedirs(generated_folder, exist_ok=True)
    if "rocksample" in env:
        num_rocks = int(env.split("_")[2])
        size = int(env.split("_")[1])
        num_obs = num_rocks * 9 + 2 + 1
        num_actions = 5 + num_rocks
    elif "safegridworld" in env:
        num_obs = 2 + 1
        num_actions = 8

    safe_mcts_obj = SafeMCTS(num_obs=num_obs, num_actions=num_actions, learning_rate=1e-3, export_path=generated_folder,
                             cost_constraint=mcts_mod.c_hat)
    if mode == "collect_samples":
        safe_mcts_obj.lagrange_mul = 0.0  # initial lagrange multiplier.
        stddev = float(sys.argv[4])  # stddev threshold / uncertainty limit to trust the safety critic.
        mcts_mod.stddev_threshold = stddev
        safe_mcts_obj.collect_samples(alpha_0=float(alpha_0), use_safety_critic=False, eps=float(eps), n_itr=0)
    elif mode == "train":
        stddev = float(stddev)
        n_max = int(n_max)
        mcts_mod.stddev_threshold = stddev
        safe_mcts_obj.learn(a_0=float(alpha_0), eps=float(eps), n_max=n_max)
    elif mode == "train_NN":
        for j in range(safe_mcts_obj.n_ensemble):  # train an ensemble of networks to predict safety critic.
            safe_mcts_obj.trainer.n_itr = int(sys.argv[4])
            trainer_mod.ckp_tag = j
            safe_mcts_obj.train_safety_critic(save_tag=f'{safe_mcts_obj.trainer.n_itr}_{j}')
    elif mode == "evaluate":
        def get_latest_checkpoint():
            files = [f for f in os.listdir(generated_folder) if f.endswith(".ckp")]
            prefix = -1
            for file in files:
                temp = int(file.split("_")[0])
                if temp > prefix:
                    prefix = temp
            return prefix
        safe_mcts_obj.lagrange_mul = 0.0  # lagrange multiplier set during evaluation.
        stddev = float(sys.argv[4])  # stddev threshold / uncertainty limit to trust the safety critic.
        mcts_mod.stddev_threshold = stddev
        safe_mcts_obj.trainer.n_itr = get_latest_checkpoint()
        safe_mcts_obj.do_planning(use_safety_critic=True,
                                  results_save=f'{safe_mcts_obj.export_path}eval_{n_sim}/',
                                  save_transitions=False, isreal=True)
