import numpy as np

def follower_best_response(follower_payoff_matrix, leader_mixed_strategy):
    """
    Compute the follower's best response in a Stackelberg game.
    
    Parameters:
    follower_payoff_matrix (np.ndarray): A 2D array where element (i, j) represents the follower's payoff 
                                         when the leader chooses action i and the follower chooses action j.
    leader_mixed_strategy (np.ndarray): A 1D array representing the leader's mixed strategy over their actions.
    
    Returns:
    int: The index of the follower's best response action.
    """
    # Number of leader and follower actions
    num_leader_actions, num_follower_actions = follower_payoff_matrix.shape
    
    # Calculate the expected payoff for each follower action
    follower_expected_payoffs = np.zeros(num_follower_actions)
    
    for j in range(num_follower_actions):
        for i in range(num_leader_actions):
            follower_expected_payoffs[j] += leader_mixed_strategy[i] * follower_payoff_matrix[i, j]
    
    # Find the index of the maximum expected payoff
    best_response_action = np.argmax(follower_expected_payoffs)
    
    return best_response_action

# Example usage
follower_payoff_matrix = np.array([
    [3, 2, 1],
    [0, 4, 3],
    [2, 1, 5]
])

leader_mixed_strategy = np.array([0.2, 0.5, 0.3])

best_response = follower_best_response(follower_payoff_matrix, leader_mixed_strategy)
print(f"The follower's best response action is: {best_response}")

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

# Example usage
follower_payoff_tensor = np.array([
    [[3, 2, 1], [2, 1, 3], [1, 3, 2]],
    [[0, 1, 0], [4, 1, 0], [3, 1, 0]],
    [[2, 1, 0], [1, 0, 1], [5, 2, 1]]
])

leader_mixed_strategy = np.array([0.2, 0.5, 0.3])
context = np.array([1.0, 2.0, 3.0])

best_response = follower_best_response_with_context(follower_payoff_tensor, leader_mixed_strategy, context)
print(f"The follower's best response action is: {best_response}")


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


# Example usage
leader_payoff_tensor = np.array([
    [[3, 2, 1], [2, 1, 3], [1, 3, 2]],
    [[0, 1, 0], [4, 1, 0], [3, 1, 0]],
    [[2, 1, 0], [1, 0, 1], [5, 2, 1]]
])

follower_payoff_tensor = np.array([
    [[3, 2, 1], [2, 1, 3], [1, 3, 2]],
    [[0, 1, 0], [4, 1, 0], [3, 1, 0]],
    [[2, 1, 0], [1, 0, 1], [5, 2, 1]]
])

leader_mixed_strategy = np.array([0.2, 0.5, 0.3])
context = np.array([1.0, 2.0, 3.0])

expected_utility = leader_expected_utility(leader_payoff_tensor, follower_payoff_tensor, leader_mixed_strategy, context)
print(f"The leader's expected utility is: {expected_utility}")