import random
import torch.nn as nn
import torch.nn.functional as F
import torch


class ReplayBuffer(object):

    def __init__(self):
        self.transitions = []
        self.train_data = []
        self.val_data = []

    def put(self, obs, action, cost, next_obs, next_action):
        if type(obs).__name__ == "list" or type(action).__name__ == "list" or type(cost).__name__ == "list" or type(next_obs).__name__ == "list" or type(next_action).__name__ == "list":
            tuple_list = zip(obs, action, cost, next_obs, next_action)
            self.transitions += tuple_list
        else:
            self.transitions.append((obs, action, cost, next_obs, next_action))  # Put a tuple of (obs, action, cost,next_obs,next_action)

    def get(self, batch_size):
        batch_samples = random.sample(self.transitions, batch_size)  # Gives batch_size samples
        obs_list, act_list, cost_list = zip(*batch_samples)
        return obs_list, act_list, cost_list

    def split_train_test(self, train_size=0.75):
        random.shuffle(self.transitions)
        idx_end = int(train_size * len(self.transitions))
        self.train_data = self.transitions[:idx_end]
        self.val_data = self.transitions[idx_end:]

    def get_batches(self, batch_size, mode="train"):
        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        if mode == "train":
            return batch(self.train_data, batch_size)
        elif mode == "test":
            return batch(self.val_data, batch_size)

    def n_items(self, mode=None):
        if mode == "train":
            return len(self.train_data)
        elif mode == "test":
            return len(self.val_data)
        else:
            return len(self.transitions)


# Start_NN
class Ensemble(nn.Module):
    def __init__(self, list_models, n_out):
        super().__init__()
        self.list_models = list_models
        self.n_out = n_out

    def forward(self, x):
        out = torch.empty((self.n_out, 0))
        if torch.cuda.is_available():
            out = out.cuda()
        for model in self.list_models:
            model_out = model(x)
            model_out = model_out.reshape(model_out.shape[0], 1)
            out = torch.cat((out, model_out), dim=1)

        out = torch.std_mean(out, dim=1, unbiased=False)
        return out


class SafetyCritic(nn.Module):
    def __init__(self, n_in, n_out):
        super(SafetyCritic, self).__init__()
        self.fcn1 = nn.Linear(n_in, 128)
        self.fcn2 = nn.Linear(128, 256)
        self.fcn3 = nn.Linear(256, 512)
        self.fcn4 = nn.Linear(512, 512)
        self.fcn5 = nn.Linear(512, 256)
        self.fcn6 = nn.Linear(256, 64)
        self.fcn7 = nn.Linear(64, n_out)

    def forward(self, x):
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        x = F.relu(self.fcn3(x))
        x = F.relu(self.fcn4(x))
        x = F.relu(self.fcn5(x))
        x = F.relu(self.fcn6(x))
        x = self.fcn7(x)
        return x


class OneStepCost(nn.Module):
    def __init__(self, n_in, n_out):
        super(OneStepCost, self).__init__()
        self.fcn1 = nn.Linear(n_in, 64)
        self.fcn2 = nn.Linear(64, n_out)

    def forward(self, x):
        x = F.relu(self.fcn1(x))
        x = F.sigmoid(self.fcn2(x))
        return x
# End_NN
