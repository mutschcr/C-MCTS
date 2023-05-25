import os

import torch as t
import torch.optim as opt
import numpy as np
from .SafetyCritic import *
import pickle
import pandas as pd

global ckp_tag


class Trainer:
    def __init__(self, num_obs, num_actions, lr, folder_save):
        self.dqn = SafetyCritic(num_obs, num_actions)
        self.dqn_target = SafetyCritic(num_obs, num_actions)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.loss_function = nn.L1Loss(reduction='sum')  # L1 loss
        self.opt = opt.Adam(self.dqn.parameters(), lr=lr)  # Optimiser
        self.experience_samples = ReplayBuffer()  # Store pretraining data
        self.folder_save = folder_save
        self.sync_after = 40
        self.n_itr = 0

    def restore_checkpoint(self, tag, cuda):
        ckp = t.load(self.folder_save + str(self.n_itr) + '_checkpoint_{:03d}.ckp'.format(tag), 'cuda' if cuda else 'cpu')
        self.dqn.load_state_dict(ckp['state_dict'])

    def add_transition(self, obs, action, cost, next_obs, next_action):
        self.experience_samples.put(obs, action, cost, next_obs, next_action)

    def store_transitions(self, path):
        with open(path + 'exp_samples.pickle', 'wb') as handle:
            pickle.dump(self.experience_samples.transitions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_transitions(self, path, train_size, balanced_split=False):
        file_paths = [f"{path}{f}/exp_samples.pickle" for f in os.listdir(path) if not f.startswith("lambda") and not f.endswith("txt") and not f.endswith("ckp") and not f.endswith("pickle") and not f.endswith(".sh") and not f.startswith("out")]
        self.clear()
        if self.n_itr > 0:
            self.n_itr -= 1
            self.restore_checkpoint(tag=ckp_tag, cuda=t.cuda.is_available())
            self.n_itr += 1
        if not balanced_split:
            for f in file_paths:
                with open(f, "rb") as handle:
                    transitions = pickle.load(handle)
                    random.shuffle(transitions)
                    split = int(train_size * len(transitions))
                    self.experience_samples.train_data += transitions[:split]
                    self.experience_samples.val_data += transitions[split:]
        else:  # only applicable for safegridworld env
            def create_empty_dict():
                unsafe_transitions = {}  # create a dictionary to place unsafe transitions in buckets.
                for i in range(8):  # x dim
                    for j in range(8):  # y dim
                        for k in range(8):  # action dim
                            unsafe_transitions[i, j, k] = []  # create an empty list that must be appended with transitions
                return unsafe_transitions

            def split_unsafe_transitions(unsafe_transitions):
                for i in range(8):  # x dim
                    for j in range(8):  # y dim
                        for k in range(8):  # action dim
                            unsafe_t = unsafe_transitions[i, j, k]
                            if len(unsafe_t) > 0:
                                random.shuffle(unsafe_t)
                                split = int(train_size * len(unsafe_t))
                                self.experience_samples.train_data += unsafe_t[:split]
                                self.experience_samples.val_data += unsafe_t[split:]

            for f in file_paths:
                with open(f, "rb") as handle:
                    transitions = pickle.load(handle)
                    if f.split("/")[-2].startswith("1_"):
                        transitions += transitions
                    elif f.split("/")[-2].startswith("2_"):
                        transitions += transitions
                        transitions += transitions
                        transitions += transitions
                        transitions += transitions
                    unsafe_transitions = create_empty_dict()
                    df = pd.DataFrame(transitions, columns=['state', 'action', 'cost', 'next_state', 'next_action'])

                    # Split safe transitions
                    if f.split("/")[-2].startswith("0_"):
                        df_safe = df[df['cost'] == 0]
                        safe_transitions = list(df_safe.itertuples(index=False, name=None))
                        random.shuffle(safe_transitions)
                        split = int(train_size * len(safe_transitions))
                        self.experience_samples.train_data += safe_transitions[:split]
                        self.experience_samples.val_data += safe_transitions[split:]

                    # Split unsafe transitions
                    df_unsafe = df[df['cost'] > 0]
                    for i, row in df_unsafe.iterrows():
                        s = row['state']
                        a = row['action']
                        unsafe_transitions[s[0], s[1], a].append(tuple(row))
                    split_unsafe_transitions(unsafe_transitions)

    def clear(self):
        self.experience_samples.transitions.clear()
        self.experience_samples.train_data.clear()
        self.experience_samples.val_data.clear()

        # Reset weights of NN
        for layer in self.dqn.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def train(self, epochs, batch_size, early_stopping_patience, cuda=False):
        def save_checkpoint(tag):
            t.save({'state_dict': self.dqn.state_dict()}, self.folder_save + str(self.n_itr) + '_checkpoint_{:03d}.ckp'.format(tag))

        def train_step(s, a, c, s_next, a_next, mode="train"):
            s = t.tensor(np.array(s), dtype=t.float)
            a = t.LongTensor(np.array(a))
            c = t.tensor(np.array(c), dtype=t.float)
            c = c.reshape(c.shape[0], 1)
            s_next = t.tensor(np.array(s_next), dtype=t.float)
            a_next = t.LongTensor(np.array(a_next))

            if cuda:
                s = s.cuda()
                c = c.cuda()
                s_next = s_next.cuda()
                a = a.cuda()
                a_next = a_next.cuda()
            q_values = self.dqn(s)
            q_values = q_values.gather(1, a.unsqueeze(1))
            next_q_values = self.dqn_target(s_next)
            next_q_values = next_q_values.gather(1, a_next.unsqueeze(1))
            expected_q_values = c + 0.95 * next_q_values
            expected_q_values = expected_q_values.clamp(0, 10)
            loss = self.loss_function(q_values, expected_q_values)  # calculate the loss
            del q_values, next_q_values, c, a_next, a, s, s_next
            if mode == "train":
                self.dqn.zero_grad()  # reset the gradients
                loss.backward()  # compute gradient by backward propagation
                self.opt.step()  # update weights

            return float(loss)  # return the loss

        def run_epoch(mode="train"):
            if mode == "train":
                self.dqn.train()
            elif mode == "test":
                self.dqn.eval()
            total_loss = 0
            n_items = self.experience_samples.n_items(mode)
            batches = self.experience_samples.get_batches(batch_size, mode)
            for i, batch in enumerate(batches):
                obs_list, act_list, cost_list, next_obs_list, next_action_list = zip(*batch)
                total_loss += train_step(obs_list, act_list, cost_list, next_obs_list, next_action_list, mode)
                if mode == "train" and i % self.sync_after == 0:
                    self.dqn_target.load_state_dict(self.dqn.state_dict())

            return total_loss / n_items

        train_loss = []
        val_loss = []
        ctr = 0
        if cuda:
            self.dqn = self.dqn.cuda()
            self.dqn_target = self.dqn_target.cuda()
            self.loss_function = self.loss_function.cuda()
        lowest_val_loss = np.inf
        while True:
            for epoch in range(epochs):  # stop by epoch number
                train_loss.append(run_epoch("train"))  # train for a epoch
                val_loss.append(run_epoch("test"))  # calculate the loss on the validation set
                if val_loss[-1] < lowest_val_loss:
                    lowest_val_loss = val_loss[-1]
                    save_checkpoint(ckp_tag)
                if len(val_loss) > 1:
                    if val_loss[-1] >= val_loss[-2]:  # validation loss does not decrease
                        ctr += 1
                    else:
                        ctr = 0
                if 0 < early_stopping_patience <= ctr:
                    break

            return train_loss, val_loss
