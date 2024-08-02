import numpy as np
import itertools, pickle, os
import matplotlib.pyplot as plt

base_dir = 'results/'
# Parameters
n = 10  # Parameter for grid generation
T = 50  # Number of iterations for Hedge algorithm
eta = 0.1  # Learning rate for Hedge algorithm
num_runs = 5

num_leader_actions = 3
num_follower_actions = 3
context_dim = 3
num_follower_types = 5

seed = 3
np.random.seed(seed)  # Fix the random seed for reproducibility

def follower_best_response_with_context(follower_payoff_tensor, leader_mixed_strategy, context):
    """
    Compute the follower's best response in a Stackelberg game considering context.
    
    Parameters:
    follower_payoff_tensor (np.ndarray): A 3D array where element (i, j, k) represents the follower's payoff 
                                         when the leader chooses action i, the follower chooses action j, 
                                         and the context is k.
    leader_mixed_strategy (np.ndarray): A 1D array representing the leader's mixed strategy over their actions.
    context (np.ndarray): A 1D array representing the context.
    
    Returns:
    int: The index of the follower's best response action.
    """
    # Number of leader actions, follower actions, and context dimensions
    num_leader_actions, num_follower_actions, num_context_dimensions = follower_payoff_tensor.shape
    
    # Calculate the expected payoff for each follower action considering the context
    follower_expected_payoffs = np.zeros(num_follower_actions)
    
    for j in range(num_follower_actions):
        for i in range(num_leader_actions):
            context_dependent_payoff = np.dot(follower_payoff_tensor[i, j], context)
            follower_expected_payoffs[j] += leader_mixed_strategy[i] * context_dependent_payoff
    
    # Find the index of the maximum expected payoff
    best_response_action = np.argmax(follower_expected_payoffs)
    
    return best_response_action

def leader_expected_utility(leader_payoff_tensor, follower_payoff_tensor, leader_mixed_strategy, context):
    """
    Compute the leader's expected utility in a Stackelberg game considering context.
    
    Parameters:
    leader_payoff_tensor (np.ndarray): A 3D array where element (i, j, k) represents the leader's payoff 
                                       when the leader chooses action i, the follower chooses action j, 
                                       and the context is k.
    follower_payoff_tensor (np.ndarray): A 3D array where element (i, j, k) represents the follower's payoff 
                                         when the leader chooses action i, the follower chooses action j, 
                                         and the context is k.
    leader_mixed_strategy (np.ndarray): A 1D array representing the leader's mixed strategy over their actions.
    context (np.ndarray): A 1D array representing the context.
    
    Returns:
    float: The expected utility of the leader.
    """
    # Get the follower's best response action
    follower_best_response = follower_best_response_with_context(follower_payoff_tensor, leader_mixed_strategy, context)
    
    # Calculate the leader's expected utility
    expected_utility = 0.0
    num_leader_actions = leader_mixed_strategy.shape[0]
    for i in range(num_leader_actions):
        context_dependent_payoff = np.dot(leader_payoff_tensor[i, follower_best_response], context)
        expected_utility += leader_mixed_strategy[i] * context_dependent_payoff
    
    return expected_utility

def compute_leader_value(context, strategy, leader_payoff_tensor, follower_payoff_tensors, follower_weight_vector):
    num_followers = follower_weight_vector.shape[0]
    leader_utility = 0
    for follower in range(num_followers):
        per_follower_utility = leader_expected_utility(leader_payoff_tensor, follower_payoff_tensors[follower], strategy, context)
        leader_utility += follower_weight_vector[follower]*per_follower_utility
    return leader_utility

# Define the Expert class
class Expert:
    def __init__(self, follower_weight_vector=None, strategy_grid=None, leader_payoff_tensor=None, follower_payoff_tensors=None, single_strategy=None):
        self.follower_weight_vector = follower_weight_vector
        self.strategy_grid = strategy_grid
        self.leader_payoff_tensor = leader_payoff_tensor
        self.follower_payoff_tensors = follower_payoff_tensors
        self.single_strategy = single_strategy

    def get_action(self, context):
        raise NotImplementedError("Subclasses must implement this method!")

# Define the Policy subclass
class Policy(Expert):
    def get_action(self, context):
        best_strategy_utility = -10
        best_strategy = None
        for strategy in self.strategy_grid:
            strategy_utility = compute_leader_value(context, strategy, self.leader_payoff_tensor, self.follower_payoff_tensors, self.follower_weight_vector)
            if strategy_utility > best_strategy_utility:
                best_strategy_utility = strategy_utility
                best_strategy = strategy
        return best_strategy

# Define the strategy subclass
class Strategy(Expert):
    def get_action(self, context):
        return self.single_strategy

# Function to generate a uniformly-spaced grid of points in a probability simplex
def generate_grid_points(n, dimension):
    """
    Generate a set of approximately uniformly spaced points in a probability simplex of given dimension.
    
    Parameters:
    n (int): The number of grid points along one dimension.
    dimension (int): The dimension of the probability simplex.
    
    Returns:
    np.ndarray: An array of shape (num_points, dimension) containing the approximately uniformly spaced points.
    """
    points = []

    # Generate all combinations of (dimension - 1) non-negative integers that sum to (n - 1)
    for comb in itertools.combinations_with_replacement(range(n + 1), dimension - 1):
        point = np.diff((0,) + comb + (n,))
        points.append(point)
    
    # Normalize to get points in the simplex
    points = np.array(points) / float(n)
    
    return points

# Hedge algorithm implementation with rewards and experts
def hedge_algorithm(experts, T, eta, leader_payoff_tensor, follower_payoff_tensors, follower_sequence, context_sequence):
    num_experts = len(experts)
    weights = np.ones(num_experts) / num_experts  # Initialize uniform weights

    # Loop through all followers, and simulate reward of each policy
    rewards = []
    for t in range(T):
        print(f"t: {t}")
        follower_payoff_tensor = follower_payoff_tensors[follower_sequence[t]]
        context = context_sequence[t]
        current_rewards = np.array([])
        for expert in experts:
            expert_strategy = expert.get_action(context)
            expert_expected_utility = leader_expected_utility(leader_payoff_tensor=leader_payoff_tensor, follower_payoff_tensor=follower_payoff_tensor, leader_mixed_strategy=expert_strategy, context=context)
            current_rewards = np.append(current_rewards, expert_expected_utility)
        rewards.append(current_rewards)
    # rewards = np.random.rand(T, num_experts)

    cumulative_rewards = np.zeros(T)

    for t in range(T):
        current_rewards = rewards[t]
        weights *= np.exp(eta * current_rewards)
        weights /= np.sum(weights)  # Normalize weights

        # Calculate the cumulative reward
        cumulative_rewards[t] = np.dot(weights, current_rewards)

    return cumulative_rewards

# Greedy algorithm implementation with rewards and experts
def greedy_algorithm(strategies, T, leader_payoff_tensor, follower_payoff_tensors, follower_sequence, context_sequence):

    num_follower_types = len(follower_payoff_tensors)
    follower_histogram = np.zeros(num_follower_types)
    realized_rewards = np.zeros(T)

    for t in range(T):
        context = context_sequence[t]
        if t==0:
            follower_dist = np.ones(num_follower_types) / num_follower_types
        else:
            follower_dist = follower_histogram / t
        
        # compute leader's strategy
        best_utility = -10
        best_strategy = None
        for strategy in strategies:

            # estimate utility of strategy
            est_strategy_utility = 0
            mixed_strategy = strategy.get_action(context)
            for follower_idx in range(num_follower_types):
                est_strategy_utility += leader_expected_utility(leader_payoff_tensor=leader_payoff_tensor, follower_payoff_tensor=follower_payoff_tensors[follower_idx], leader_mixed_strategy=mixed_strategy, context=context) * follower_dist[follower_idx]

            if est_strategy_utility > best_utility:
                best_utility = est_strategy_utility
                best_strategy = mixed_strategy
        
        # get utility of playing best_strategy
        reward = leader_expected_utility(leader_payoff_tensor=leader_payoff_tensor, follower_payoff_tensor=follower_payoff_tensors[follower_sequence[t]], leader_mixed_strategy=best_strategy, context=context)
        
        # Calculate the realized reward
        realized_rewards[t] = reward

        # Update follower histogram
        follower_histogram[follower_sequence[t]] += 1

    return realized_rewards

def plotting(context_str, follower_str):

    # unpickle
    fname = base_dir + f"baseline_context={context_str}_follower={follower_str}_n={n}_T={T}_eta={eta}_num_runs={num_runs}_num_leader_actions={num_leader_actions}_num_follower_actions={num_follower_actions}_context_dim_{context_dim}_num_follower_types{num_follower_types}.pkl"
    baseline_run_list = pickle.load(open(fname,'rb'))

    fname = base_dir + f"policy_context={context_str}_follower={follower_str}_n={n}_T={T}_eta={eta}_num_runs={num_runs}_num_leader_actions={num_leader_actions}_num_follower_actions={num_follower_actions}_context_dim_{context_dim}_num_follower_types{num_follower_types}.pkl"
    policy_run_list = pickle.load(open(fname,'rb'))

    fname = base_dir + f"greedy_context={context_str}_follower={follower_str}_n={n}_T={T}_eta={eta}_num_runs={num_runs}_num_leader_actions={num_leader_actions}_num_follower_actions={num_follower_actions}_context_dim_{context_dim}_num_follower_types{num_follower_types}.pkl"
    greedy_run_list = pickle.load(open(fname,'rb'))

    # compute mean + std for each
    # Stack the arrays into a 2D numpy array
    stacked_baseline = np.vstack(baseline_run_list)
    stacked_policy = np.vstack(policy_run_list)
    stacked_greedy = np.vstack(greedy_run_list)
    
    # Compute the element-wise mean
    baseline_mean = np.mean(stacked_baseline, axis=0)
    policy_mean = np.mean(stacked_policy, axis=0)
    greedy_mean = np.mean(stacked_greedy, axis=0)
    
    # Compute the element-wise standard deviation
    baseline_std = np.std(stacked_baseline, axis=0)
    policy_std = np.std(stacked_policy, axis=0)
    greedy_std = np.std(stacked_greedy, axis=0)

    # Plot cumulative reward as a function of time
    t_range = list(range(1, T + 1))

    plt.plot(t_range, baseline_mean, label="baseline")
    plt.fill_between(t_range, baseline_mean - baseline_std, baseline_mean + baseline_std, alpha=0.2)

    plt.plot(t_range, greedy_mean, label="Algorithm 1")
    plt.fill_between(t_range, greedy_mean - greedy_std, greedy_mean + greedy_std, alpha=0.2)

    plt.plot(t_range, policy_mean, label="Algorithm 2")
    plt.fill_between(t_range, policy_mean - policy_std, policy_mean + policy_std, alpha=0.2)

    plt.xlabel('Time')
    plt.ylabel('Cumulative Reward')
    if context_str == 'stoch':
        context_str = 'stochastic'
    elif context_str == 'adv':
        context_str = 'non-stochastic'
    if follower_str == 'stoch':
        follower_str = 'stochastic'
    elif follower_str == 'adv':
        follower_str = 'non-stochastic'
    plt.title(f'Contexts: {context_str}, Followers: {follower_str}')
    plt.legend()
    plt.show()

def sweep(adv_contexts = False, adv_followers = False):
    baseline_run_list = []
    policy_run_list = []
    greedy_run_list = []
    for run in range(num_runs):
        if adv_followers:
            follower_sequence = generate_adv_followers()
            follower_str = 'adv'
        else:
            # generate random sequence of followers
            follower_sequence = generate_random_followers()
            follower_str = 'stoch'
        print("generated follower sequence")

        if adv_contexts:
            # generate adversarial sequence of contexts
            context_sequence = generate_adv_contexts()
            context_str = 'adv'
        else:
            # generate random sequence of contexts
            context_sequence = generate_random_contexts()
            context_str = 'stoch'
        print("generated context sequence")


        # Run Hedge algorithm
        baseline_rewards = hedge_algorithm(point_experts, T, eta, leader_payoff_tensor, follower_payoff_tensors, follower_sequence, context_sequence)
        print("ran Hedge on baseline")
        cumulative_baseline_rewards = np.cumsum(baseline_rewards)

        policy_rewards = hedge_algorithm(policy_experts, T, eta, leader_payoff_tensor, follower_payoff_tensors, follower_sequence, context_sequence)
        print("ran Hedge on policies")
        cumulative_policy_rewards = np.cumsum(policy_rewards)

        # run Greedy algorithm
        greedy_rewards = greedy_algorithm(strategies=point_experts, T=T, leader_payoff_tensor=leader_payoff_tensor, follower_payoff_tensors=follower_payoff_tensors, follower_sequence=follower_sequence, context_sequence=context_sequence)
        print("ran Greedy algorithm")
        cumulative_greedy_rewards = np.cumsum(greedy_rewards)

        baseline_run_list.append(cumulative_baseline_rewards)
        policy_run_list.append(cumulative_policy_rewards)
        greedy_run_list.append(cumulative_greedy_rewards)
        print(f"Run {run} completed")
    
    fname = base_dir + f"baseline_context={context_str}_follower={follower_str}_n={n}_T={T}_eta={eta}_num_runs={num_runs}_num_leader_actions={num_leader_actions}_num_follower_actions={num_follower_actions}_context_dim_{context_dim}_num_follower_types{num_follower_types}.pkl"
    pickle.dump(baseline_run_list, open(fname, 'wb'))

    fname = base_dir + f"policy_context={context_str}_follower={follower_str}_n={n}_T={T}_eta={eta}_num_runs={num_runs}_num_leader_actions={num_leader_actions}_num_follower_actions={num_follower_actions}_context_dim_{context_dim}_num_follower_types{num_follower_types}.pkl"
    pickle.dump(policy_run_list, open(fname, 'wb'))

    fname = base_dir + f"greedy_context={context_str}_follower={follower_str}_n={n}_T={T}_eta={eta}_num_runs={num_runs}_num_leader_actions={num_leader_actions}_num_follower_actions={num_follower_actions}_context_dim_{context_dim}_num_follower_types{num_follower_types}.pkl"
    pickle.dump(greedy_run_list, open(fname, 'wb'))

    # plotting(context_str=context_str, follower_str=follower_str)

def generate_random_contexts():
    return [np.random.uniform(-1, 1, size=context_dim) for _ in range(T)]

def generate_adv_contexts():
    context1 = np.zeros(context_dim)
    context2 = np.ones(context_dim)
    context3 = -1*np.ones(context_dim)
    context_list = [context1, context2, context3, context1]
    contexts = [context for context in context_list for _ in range(T//4)]
    contexts.append(context1)
    contexts.append(context2)
    return contexts

def generate_random_followers():
    return [np.random.randint(0, num_follower_types - 1) for _ in range(T)]

def generate_adv_followers():
    follower_list = []
    for t in range(T//num_follower_types):
        for follower_idx in range(num_follower_types):
            follower_list.append(follower_idx)
    return follower_list

if __name__ == '__main__':
    # shape = (context_dim, num_leader_actions, num_follower_actions)
    # follower_payoff_tensors = [np.random.uniform(-1, 1, size=shape) for _ in range(num_follower_types)]

    # leader_payoff_tensor = np.random.uniform(-1, 1, size=shape)

    # # Generate grid points
    # grid_points = generate_grid_points(n, dimension=num_leader_actions)

    # # Instantiate experts as gridpoints
    # point_experts = [Strategy(single_strategy=point) for point in grid_points]

    # # Generate sets of weights (one for each policy)
    # follower_weight_list = generate_grid_points(n, dimension=num_follower_types)
    # # Instantiate experts as policies
    # policy_experts = [Policy(follower_weight_vector=follower_weight_vector, strategy_grid=grid_points, leader_payoff_tensor=leader_payoff_tensor, follower_payoff_tensors=follower_payoff_tensors) for follower_weight_vector in follower_weight_list]
    # print("Instantiated policies")

    # sweep(adv_contexts=False, adv_followers=True)
    # sweep(adv_contexts=True, adv_followers=False)
    # sweep(adv_contexts=False, adv_followers=False)

    plotting(context_str='stoch', follower_str='adv')
    plotting(context_str='adv', follower_str='stoch')
    plotting(context_str='stoch', follower_str='stoch')