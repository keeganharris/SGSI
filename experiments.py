import numpy as np
import itertools
import matplotlib.pyplot as plt

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

def compute_leader_utility(context, strategy, leader_payoff_tensor, follower_payoff_tensors, follower_weight_vector):
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
        for strat in self.strategy_grid:
            strategy_utility = compute_leader_utility(context, strat, self.leader_payoff_tensor, self.follower_payoff_tensors, self.follower_weight_vector)
            if strategy_utility > best_strategy_utility:
                best_strategy_utility = strategy_utility
                best_strategy = strat
        return best_strategy

# Define the strategy subclass
class Strategy(Expert):
    def get_action(self, context):
        return self.single_strategy

# Function to generate a uniformly-spaced grid of points in a probability simplex
def generate_grid_points(n, dimension=5):
    # Generate all possible combinations of (n+1) non-negative integers that sum to n
    points = [point for point in itertools.combinations_with_replacement(range(n+1), 5)]
    # Convert each combination to a point in the simplex by normalizing the sum to 1
    grid_points = []
    for point in points:
        remaining_sum = n - sum(point)
        for i in range(dimension):
            new_point = list(point)
            new_point[i] += remaining_sum
            grid_points.append(np.array(new_point) / n)
    return np.array(grid_points)

# Hedge algorithm implementation with rewards and experts
def hedge_algorithm(experts, num_iterations, eta):
    num_experts = len(experts)
    weights = np.ones(num_experts) / num_experts  # Initialize uniform weights

    # Simulate some rewards for demonstration (random rewards)
    rewards = np.random.rand(num_iterations, num_experts)

    cumulative_rewards = np.zeros(num_iterations)

    for t in range(num_iterations):
        current_rewards = rewards[t]
        weights *= np.exp(eta * current_rewards)
        weights /= np.sum(weights)  # Normalize weights

        # Calculate the cumulative reward
        cumulative_rewards[t] = np.dot(weights, current_rewards)

    return cumulative_rewards

if __name__ == '__main__':
    # Parameters
    n = 10  # Parameter for grid generation
    num_iterations = 1000  # Number of iterations for Hedge algorithm
    eta = 0.1  # Learning rate for Hedge algorithm

    num_leader_actions = 3
    num_follower_actions = 3
    context_dim = 3

    leader_payoff_tensor = np.array([
    [[3, 2, 1], [2, 1, 3], [1, 3, 2]],
    [[0, 1, 0], [4, 1, 0], [3, 1, 0]],
    [[2, 1, 0], [1, 0, 1], [5, 2, 1]]
    ])
    follower_payoff_tensor1 = np.array([
    [[3, 2, 1], [2, 1, 3], [1, 3, 2]],
    [[0, 1, 0], [4, 1, 0], [3, 1, 0]],
    [[2, 1, 0], [1, 0, 1], [5, 2, 1]]
    ])
    follower_payoff_tensor2 = np.array([
    [[3, 2, 1], [2, 1, 3], [1, 3, 2]],
    [[0, 1, 0], [4, 1, 0], [3, 1, 0]],
    [[2, 1, 0], [1, 0, 1], [5, 2, 1]]
    ])
    follower_payoff_tensor_list = [follower_payoff_tensor1, follower_payoff_tensor2]
    num_follower_types = len(follower_payoff_tensor_list)

    # Generate grid points
    grid_points = generate_grid_points(n, dimension=num_leader_actions)
    # Instantiate experts as gridpoints
    point_experts = [Strategy(single_strategy=point) for point in grid_points]

    # Generate sets of weights (one for each policy)
    follower_weight_list = generate_grid_points(n, dimension=num_follower_types)
    # Instantiate experts as policies
    policy_experts = [Policy(follower_weight_vector=follower_weight_vector, strategy_grid=grid_points, leader_payoff_tensor=leader_payoff_tensor, follower_payoff_tensors=follower_payoff_tensor_list) for follower_weight_vector in follower_weight_list]

    # Run Hedge algorithm
    cumulative_rewards_baseline = hedge_algorithm(point_experts, num_iterations, eta)
    cumulative_rewards_policies = hedge_algorithm(policy_experts, num_iterations, eta)

    # Plot cumulative reward as a function of time
    plt.plot(np.cumsum(cumulative_rewards_baseline), label="baseline")
    plt.plot(np.cumsum(cumulative_rewards_policies), label="policies")
    plt.xlabel('Time')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward as a Function of Time')
    plt.legend()
    plt.show()