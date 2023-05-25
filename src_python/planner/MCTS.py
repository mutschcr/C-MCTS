import numpy as np
import copy
import torch as t

# global variable
var_lambda = np.array(8, dtype=float)  # lagrangian multiplier
tree_depth = 0
c_hat = 1
stddev_threshold = 2
C = 0
gamma = 1.0


# TODO: Keep an eye on how seeding is affecting the random module in numpy.

class VNode:
    def __init__(self, state, parent=None):
        self.n_actions = None
        self._state = state
        self.parent = parent
        self.children = {}
        self.is_expanded = False
        self.child_P = None  # priors
        self.child_Q = None  # Q value
        self.child_Qc = None  # MC cost estimates
        self.child_Qc_NN = None  # NN cost estimates
        self.child_N = None  # n_visits
        self.child_R = None  # one-step reward
        self.child_C = None  # one-step costs
        self.V = np.array([0., 0.])
        self.N = 0

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, s):
        self._state = s

    def isterminal(self):
        return self._state[-1]

    def Qavgs(self):
        out = self.child_Q / (self.child_N+1)
        isNaN = np.any(np.isnan(out))
        return out

    def Qcavgs(self):
        out = self.child_Qc / (self.child_N+1)
        isNaN = np.any(np.isnan(out))
        return out

    def UCBs(self):
        compute_mask = self.child_N != 0
        logN = np.log(np.ones(compute_mask.sum()) * self.N + 1)
        out = np.ones(self.n_actions) * 1e10
        out[compute_mask] = self.child_P[compute_mask] * np.sqrt(logN / self.child_N[compute_mask])
        return out

    def best_child_idx(self, ucb=True):
        global tree_depth, c_hat

        # calculate Q values of child nodes with UCB bounds
        f = self.Qavgs() - var_lambda * self.Qcavgs()
        f_plus = copy.deepcopy(f)
        if ucb:
            f_plus += self.UCBs()

        # calculate bias term to form the set 'A': set of best actions
        # Note: Implemented based on the CC-POMCP paper, CC-POMCP code base has deviations in the formula
        actionBias = np.zeros(len(self.child_N))
        compute_mask = self.child_N != 0
        actionBias[compute_mask] = np.sqrt(np.log(self.child_N[compute_mask]) / self.child_N[compute_mask])
        biasconstant = np.exp(-tree_depth) * 0.1
        best_idx = np.random.choice(np.flatnonzero(f_plus == f_plus.max()))  # idx with highest Q_plus value
        threshold = (actionBias[best_idx] + actionBias) * biasconstant

        greedy_indices = np.flatnonzero(np.abs(f_plus - f_plus[best_idx]) <= threshold)  # set of greedy actions
        greedy_Qc = self.Qcavgs()[greedy_indices]  # costs corresponding to greedy actions
        minCost = greedy_Qc.min()
        minCost_idx = greedy_indices[greedy_Qc.argmin()]
        maxCost = greedy_Qc.max()
        maxCost_idx = greedy_indices[greedy_Qc.argmax()]

        if maxCost <= c_hat:
            minCost_idx = maxCost_idx
            probMinCost = probMaxCost = 1

        elif minCost >= c_hat:
            minCost_idx = minCost_idx
            probMinCost = probMaxCost = 1

        else:
            probMinCost = (c_hat - maxCost) / (minCost - maxCost)
            probMaxCost = 1 - probMinCost

        return probMinCost, probMaxCost, minCost_idx, maxCost_idx

    def select_leaf(self):
        current = self
        global tree_depth, C, gamma
        while current.is_expanded:
            pmin, pmax, idx_min, idx_max = current.best_child_idx(ucb=True)
            if np.random.uniform(0, 1) < pmin:
                child_idx = idx_min
            else:
                child_idx = idx_max
            C += gamma * current.child_C[child_idx]
            current = current.children[child_idx].child
            tree_depth += 1
            gamma *= 0.95
        return current

    def expand(self, states, actions, rewards_costs, priors, Qc_NN=None):
        self.is_expanded = True
        self.n_actions = len(actions)
        self.child_P = np.zeros([self.n_actions], dtype=np.float32)  # priors
        self.child_Q = np.zeros([self.n_actions], dtype=np.float32)  # Q value
        self.child_Qc = np.zeros([self.n_actions], dtype=np.float32)  # cost estimate
        self.child_N = np.zeros([self.n_actions], dtype=np.float32)  # n_visits
        self.child_R = np.zeros([self.n_actions], dtype=np.float32)  # one-step reward
        self.child_C = np.zeros([self.n_actions], dtype=np.float32)  # one-step costs

        for i, a in enumerate(actions):
            self.add_child(i, a, states[i])
            self.child_P[i] = priors[i]
            self.child_R[i] = rewards_costs[i][0]
            self.child_C[i] = rewards_costs[i][1]

        if Qc_NN is not None:
            self.child_Qc_NN = Qc_NN

    def add_child(self, idx, a, state=None):
        self.children[idx] = QNode(a, idx, state, self)

    def backup(self, value_backup, discount_factor):
        current = self
        current.N += 1
        global tree_depth
        while current.parent is not None:
            QNode_current = current.parent
            value_backup[0] = QNode_current.R + discount_factor * value_backup[0]
            value_backup[1] = QNode_current.C + discount_factor * value_backup[1]
            QNode_current.Q += value_backup[0]
            QNode_current.Qc += value_backup[1]
            QNode_current.N += 1
            current = QNode_current.parent
            current.N += 1
            tree_depth -= 1

        current.V = ((current.N - 1) * current.V + value_backup) / current.N


class QNode:
    def __init__(self, action, idx, state=None, parent=None):
        self.action = action
        self.idx = idx
        self.parent = parent
        self._child = VNode(state, self)

    @property
    def a(self):
        return self.action

    @property
    def N(self):
        return self.parent.child_N[self.idx]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.idx] = value

    @property
    def Q(self):
        return self.parent.child_Q[self.idx]

    @Q.setter
    def Q(self, value):
        self.parent.child_Q[self.idx] = value

    @property
    def R(self):
        return self.parent.child_R[self.idx]

    @R.setter
    def R(self, reward):
        self.parent.child_R[self.idx] = reward

    @property
    def C(self):
        return self.parent.child_C[self.idx]

    @C.setter
    def C(self, cost):
        self.parent.child_C[self.idx] = cost

    @property
    def Qc(self):
        return self.parent.child_Qc[self.idx]

    @Qc.setter
    def Qc(self, value):
        self.parent.child_Qc[self.idx] = value

    @property
    def child(self):
        return self._child

    @child.setter
    def child(self, state):
        self._child = VNode(state, self)


class MCTS:
    def __init__(self, simulator, n_actions, n_simulations, n_rollouts, rollout_depth, discount, cost_constraint, verbose=0, safety_critic=None):
        global c_hat
        self.simulator = simulator
        self.n_simulations = n_simulations
        self.n_rollouts = n_rollouts
        self.rollout_depth = rollout_depth
        self.discount = discount
        self.cost_constraint = c_hat = cost_constraint
        self.prior = 20
        self.n_actions = n_actions
        self.lambdaMax = 30
        self.n_pruned = np.zeros(50)
        self.depth_max = -1e10
        self.C = 0
        self.verbose = verbose
        self.rollout_action_distribution = np.zeros((self.rollout_depth, self.n_actions))
        if safety_critic is None:
            self.use_NN = False
            self.safety_critic = None
        else:
            self.use_NN = True
            self.safety_critic = safety_critic
            if t.cuda.is_available():
                self.safety_critic = self.safety_critic.cuda()
                for model in self.safety_critic.list_models:
                    model = model.cuda()

    def set_cost_constraint(self, cost, probActTaken, otherCost):
        self.cost_constraint = (self.cost_constraint - probActTaken * cost - (1 - probActTaken) * otherCost) / (self.discount * probActTaken)
        if self.cost_constraint < 0:
            self.cost_constraint = 0

    def get_cost_estimate(self, state, action):
        state = t.tensor(state, dtype=t.float)
        if t.cuda.is_available():
            state = state.cuda()
        Qc = self.safety_critic.forward(state)
        if type(action).__name__ == "ndarray":
            action = t.LongTensor(action)
            if t.cuda.is_available():
                action = action.cuda()
            std_dev = Qc[0].gather(0, action)
            mean = Qc[1].gather(0, action)
        else:
            std_dev = Qc[0][action]
            mean = Qc[1][action]
        if t.cuda.is_available():
            mean = mean.cpu().detach().numpy()
            std_dev = std_dev.cpu().detach().numpy()
        else:
            mean = mean.detach().numpy()
            std_dev = std_dev.detach().numpy()

        if type(mean).__name__ == "ndarray":
            mean[std_dev > stddev_threshold] = -1e-10
        else:
            if std_dev > stddev_threshold:
                mean = -1e-10
        mean = np.clip(mean, a_min=0, a_max=20)
        return mean

    def rollout(self, state):
        self.simulator.set_state(state)
        V = np.array([0., 0.])
        global tree_depth
        V = self.simulator.rollout(tree_depth)
        return V

    def expand(self, leaf: VNode):
        global C, gamma
        if not leaf.isterminal():
            self.simulator.set_state(leaf.state)
            legal_actions = self.simulator.get_legalactions()

            # prune branches of the tree using a safety critic.
            if self.use_NN:
                qc_values = np.round(self.get_cost_estimate(leaf.state, legal_actions),1)
                qc_values = C + gamma * qc_values
                if tree_depth >= 0 and np.count_nonzero(qc_values <= self.cost_constraint) == 0:
                    legal_actions = legal_actions[qc_values == qc_values.min()]
                else:
                    legal_actions = legal_actions[qc_values <= self.cost_constraint]

            n_actions = len(legal_actions)
            if n_actions == 0:
                return False
            states = []
            actions = []
            rc_list = []
            Qc_NN = None
            priors = np.ones(n_actions) * self.prior
            for i in range(n_actions):  # TODO: implement this in c++ and call: (eliminate python loops)
                rc = self.simulator.step(legal_actions[i])
                s = self.simulator.get_state()
                states.append(s)
                actions.append(legal_actions[i])
                rc_list.append(rc)
                self.simulator.set_state(leaf.state)

            if self.use_NN:
                Qc_NN = qc_values

            leaf.expand(states, actions, rc_list, priors, Qc_NN)
            return True
        return True

    def get_best_action(self, state, ilambda=8):
        self.n_pruned = np.zeros(50)
        self.depth_max = -1e10

        def get_node_estimate(node: VNode):
            # get estimates for back up
            value_estimate = np.array([0., 0.])
            self.rollout_action_distribution = np.zeros(self.rollout_action_distribution.shape)
            if not node.isterminal():
                for i in range(self.n_rollouts):
                    value_estimate += self.rollout(node.state)
                value_estimate /= self.n_rollouts  # average over 'n' rollouts
                self.rollout_action_distribution /= self.n_rollouts
                if self.verbose > 0:
                    print(self.rollout_action_distribution)

            return value_estimate

        global var_lambda, tree_depth, C, gamma
        var_lambda = ilambda
        tree_depth = 0
        C = 0.0
        gamma = 1.0
        root = VNode(state)
        self.simulator.set_state(state)  # set the root node

        # get a rough value estimate of the root and back up.
        value_estimate = get_node_estimate(root)
        root.backup(value_estimate, self.discount)

        self.expand(root)  # expand the root node

        for n_sim in range(self.n_simulations):
            # Selection : Traverse from the root node to the leaf node
            C = 0.0
            gamma = 1.0
            leaf = root.select_leaf()

            if tree_depth > self.depth_max:
                self.depth_max = tree_depth

            # do rollouts from the leaf node and get estimates for back up
            value_estimate = get_node_estimate(leaf)

            # Expansion : Fully expand the leaf node
            expanded = self.expand(leaf)

            if not expanded:  # all actions are unsafe.
                value_estimate[1] = 30  # back up a high cost, if a state is a dead-end i.e. all actions are unsafe.

            # Backup from the leaf node
            leaf.backup(value_estimate, self.discount)

        # best policy
        probMin, probMax, idx_min, idx_max = root.best_child_idx(ucb=False)
        a_min = root.children[idx_min].a
        a_max = root.children[idx_max].a
        cost_min = root.Qcavgs()[idx_min]
        cost_max = root.Qcavgs()[idx_max]

        policy = [{'action': a_min, 'p': probMin, 'cost': cost_min},
                  {'action': a_max, 'p': probMax, 'cost': cost_max}]

        return policy, root, self.n_pruned, self.depth_max
