import MCTS as mcts_py
from MCTS import MCTS
import numpy as np


class test_simulator:
    def __init__(self, n_actions, start_state=None, discount=0.95, max_depth=10):
        self.n_actions = n_actions
        self.current_state = start_state
        self.tdepth = 0
        self.discount = discount
        self.max_depth = max_depth

    def get_legalactions(self):
        return np.linspace(0, self.n_actions - 1, self.n_actions, dtype=int)

    def get_state(self):
        return self.current_state

    def set_state(self, state):
        self.current_state = state

    def step(self, action):
        if action == 0:
            return [1, 1]
        else:
            return [0, 0]

    def mean(self):
        discount = 1.0
        rc = np.array([0., 0.])
        for _ in range(self.max_depth):
            rc += np.array([discount / self.n_actions, discount / self.n_actions])
            discount *= self.discount

        return rc

    def optimal(self):
        discount = 1.0
        rc = np.array([0., 0.])
        for _ in range(self.max_depth):
            rc += np.array([discount, discount])
            discount *= self.discount
        return rc


def run_tests():
    test_greedy()
    test_UCB()
    test_rollouts()
    test_search()


def test_greedy():
    n_actions = 4
    sim = test_simulator(n_actions, [None, 0])
    mcts_planner = MCTS(simulator=sim, n_actions=n_actions, n_simulations=4, n_rollouts=1, rollout_depth=0, discount=0.95, cost_constraint=1)

    # Test 1: sensitivity to lagrange multipliers
    def test_step1(self, action):
        if action == 0:
            return np.array([1., 1.])
        else:
            return np.array([0., 0.])

    mcts_planner.simulator.step = test_step1.__get__(mcts_planner.simulator, type(mcts_planner.simulator))
    p1, _, _ = mcts_planner.get_best_action([None, 0], 0)  # Q = [1,0,0,0]
    p2, _, _ = mcts_planner.get_best_action([None, 0], 2)  # Q = [-1,0,0,0]
    assert (p1[0]['action'] == p1[1]['action'] == 0 and p1[0]['p'] == p1[1]['p'] == 1 and
            p2[0]['action'] == p2[1]['action'] != 0 and p2[0]['p'] == p2[1]['p'] == 1)

    # Test 2: Discriminate actions based on cost. reward:0,cost:1.
    def test_step2(self, action):
        if action == 0:
            return np.array([0., 0.])
        else:
            return np.array([0., 1.])

    mcts_planner.simulator.step = test_step2.__get__(mcts_planner.simulator, type(mcts_planner.simulator))
    p1, _, _ = mcts_planner.get_best_action([None, 0], 1)
    assert (p1[0]['action'] == p1[1]['action'] == 0 and p1[0]['p'] == p1[1]['p'] == 1)

    # Test 3: Stochastic policy check
    mcts_planner.cost_constraint = 0.75
    mcts_py.c_hat = 0.75

    def test_step3(self, action):
        if action == 0:
            return np.array([0.375, 0.375])
        elif action == 1:
            return np.array([1., 1.])
        else:
            return np.array([-1e10, 1e10])  # sub-optimal actions

    mcts_planner.simulator.step = test_step3.__get__(mcts_planner.simulator, type(mcts_planner.simulator))
    p1, _, _ = mcts_planner.get_best_action([None, 0], 1)
    assert (p1[0]['action'] == 0 and p1[0]['p'] == 0.4 and p1[0]['cost'] == 0.375 and
            p1[1]['action'] == 1 and p1[1]['p'] == 0.6 and p1[1]['cost'] == 1)


def test_UCB():
    n_actions = 5
    sim = test_simulator(n_actions, [None, 0])
    mcts_planner = MCTS(simulator=sim, n_actions=n_actions, n_simulations=n_actions, n_rollouts=1, rollout_depth=0, discount=0.95, cost_constraint=1)

    # Test 1: with equal value, action with lowest count is selected
    def test_step1(self, action):
        return np.array([0., 0.])

    mcts_planner.simulator.step = test_step1.__get__(mcts_planner.simulator, type(mcts_planner.simulator))
    _, root, _ = mcts_planner.get_best_action([None, 0], 1)
    list_N = []
    for a in range(n_actions):
        if a == 3:
            list_N.append(99)
        else:
            list_N.append(100 + a)
    root.child_N = np.array(list_N)  # override count value such that action '3' is least explored
    root.N = root.child_N.sum()
    root.child_Q *= root.child_N
    root.child_Qc *= root.child_N
    mcts_py.tree_depth = 1
    p1, p2, idx1, idx2 = root.best_child_idx(ucb=True)
    a1 = root.children[idx1].a
    a2 = root.children[idx2].a
    assert (p1 == p2 == 1 and a1 == a2 == 3)

    # Test 2: with high counts, action with highest value is selected
    def test_step2(self, action):
        if action == 3:
            return np.array([1., 1.])
        else:
            return np.array([0., 0.])

    mcts_planner.simulator.step = test_step2.__get__(mcts_planner.simulator, type(mcts_planner.simulator))
    _, root, _ = mcts_planner.get_best_action([None, 0], 0.3)
    list_N = []
    for a in range(n_actions):
        if a == 3:
            list_N.append(99 + 5)
        else:
            list_N.append(100 + 5 - a)
    root.child_N = np.array(list_N)
    root.N = root.child_N.sum()
    root.child_Q *= root.child_N
    root.child_Qc *= root.child_N
    mcts_py.tree_depth = 1
    p1, p2, idx1, idx2 = root.best_child_idx(ucb=True)
    a1 = root.children[idx1].a
    a2 = root.children[idx2].a
    assert (p1 == p2 == 1 and a1 == a2 == 3)

    # Test 3: action with low value and low count beats actions with high counts.
    def test_step3(self, action):
        return np.array([1., 1.])

    mcts_planner.simulator.step = test_step3.__get__(mcts_planner.simulator, type(mcts_planner.simulator))
    _, root, _ = mcts_planner.get_best_action([None, 0], 0.3)
    list_N = []
    for a in range(n_actions):
        if a == 3:
            list_N.append(1)
        else:
            list_N.append(100 + a)
    root.child_N = np.array(list_N)  # override count value such that action '3' is least explored
    root.N = root.child_N.sum()
    root.child_Q *= root.child_N
    root.child_Qc *= root.child_N
    mcts_py.tree_depth = 1
    p1, p2, idx1, idx2 = root.best_child_idx(ucb=True)
    a1 = root.children[idx1].a
    a2 = root.children[idx2].a
    assert (p1 == p2 == 1 and a1 == a2 == 3)

    # Test 4: actions with zero count is always selected.
    def test_step4(self, action):
        if action == 3:
            return np.array([0., 0.])
        else:
            return np.array([100., 1.])

    mcts_planner.simulator.step = test_step4.__get__(mcts_planner.simulator, type(mcts_planner.simulator))
    _, root, _ = mcts_planner.get_best_action([None, 0], 0.3)
    list_N = []
    for a in range(n_actions):
        if a == 3:
            list_N.append(0)
        else:
            list_N.append(100 + a)
    root.child_N = np.array(list_N)  # override count value such that action '3' is least explored
    root.N = root.child_N.sum()
    root.child_Q *= root.child_N
    root.child_Qc *= root.child_N
    mcts_py.tree_depth = 1
    p1, p2, idx1, idx2 = root.best_child_idx(ucb=True)
    a1 = root.children[idx1].a
    a2 = root.children[idx2].a
    assert (p1 == p2 == 1 and a1 == a2 == 3)
    _, root, _ = mcts_planner.get_best_action([None, 0], 0.3)
    list_N.reverse()  # now action '1' has zero count
    root.child_N = np.array(list_N)  # override count value such that action '1' is least explored
    root.N = root.child_N.sum()
    root.child_Q *= root.child_N
    root.child_Qc *= root.child_N
    p1, p2, idx1, idx2 = root.best_child_idx(ucb=True)
    a1 = root.children[idx1].a
    a2 = root.children[idx2].a
    assert (p1 == p2 == 1 and a1 == a2 == 1)


def test_rollouts():
    def rollouts(n_rollouts):
        n_actions = 2
        max_depth = 10
        n_rollouts = n_rollouts
        sim = test_simulator(n_actions, [None, 0], discount=0.95, max_depth=max_depth)
        mcts_planner = MCTS(simulator=sim, n_actions=n_actions, n_simulations=0, n_rollouts=n_rollouts, rollout_depth=max_depth, discount=0.95, cost_constraint=1, verbose=1)

        # Test: state value estimate of the leaf node improves with n_rollouts.
        def test_step(self, action):
            if self.tdepth < max_depth and action == 0:
                return np.array([1., 1.])
            else:
                return np.array([0., 0.])
            self.tdepth += 1

        mcts_planner.simulator.step = test_step.__get__(mcts_planner.simulator, type(mcts_planner.simulator))
        _, root, _ = mcts_planner.get_best_action([None, 0], 0.3)
        print(f"------Test rollouts (number of rollouts: {n_rollouts}, rollout depth: {max_depth})-----------")
        print(f"estimate: {root.V}")
        print(f"actual: {sim.mean()}")
        print(f"error: {np.abs(root.V - sim.mean())}")
        return np.abs(root.V - sim.mean())

    # Test: value estimates become better with higher 'n_rollouts'.
    error_list = []
    n_rollouts = [10, 100, 1000]
    error_prev = np.array([1e10, 1e10])
    for n in n_rollouts:
        for i in range(20):
            error_list.append(rollouts(n))
        # print(np.array(error_list))
        error_cur = np.array(error_list).mean(0)
        assert ((error_cur < error_prev).all())
        error_prev = error_cur


def test_search():
    n_actions = 3
    max_depth = 10
    sim = test_simulator(n_actions, [None, 0], discount=0.95, max_depth=max_depth)
    mcts_planner = MCTS(simulator=sim, n_actions=n_actions, n_simulations=40, n_rollouts=1, rollout_depth=max_depth, discount=0.95, cost_constraint=1)

    def test_step(self, action):
        if self.tdepth < max_depth and action == 0:
            return np.array([1., 1.])
        else:
            return np.array([0., 0.])
        self.tdepth += 1

    mcts_planner.simulator.step = test_step.__get__(mcts_planner.simulator, type(mcts_planner.simulator))

    # Test 1: lagrange multiplier is getting set correctly.
    mcts_py.var_lambda = np.random.uniform(0, 30)
    ilambda = 0.3
    _, root, _ = mcts_planner.get_best_action([None, 0], ilambda=ilambda)
    assert (mcts_py.var_lambda == ilambda)

    # Test 2: Search improves with better confidence for higher number of simulations.
    # TODO: is the performance obtained acceptable (i.e. Q value of best action / ratio of 'N' across child nodes in root)
    n_simulations = [40, 200, 400, 800]
    n_actions = 3
    max_depth = 10
    prob_prev = -1e10
    for n_sim in n_simulations:
        mcts_ = MCTS(simulator=sim, n_actions=n_actions, n_simulations=n_sim, n_rollouts=1, rollout_depth=max_depth, discount=0.95, cost_constraint=1)
        prob_N_list = []
        for n_runs in range(20):
            _, root, _ = mcts_.get_best_action([None, 0], ilambda=ilambda)
            prob_N = root.child_N / root.child_N.sum()
            prob_N_list.append(prob_N)
        print(f"-------n_simulations={n_sim}---------------")
        print(f"probability to choose best action: {np.array(prob_N_list).mean(0)[0]} +/- {np.array(prob_N_list).var(0)[0]}")
        prob_cur = np.array(prob_N_list).mean(0)[0]
        assert (prob_cur > prob_prev)
        prob_prev = prob_cur


run_tests()
