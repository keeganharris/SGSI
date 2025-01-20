import numpy as np
import matplotlib.pyplot as plt
import pickle

# Global variables for easy configurability
CONTEXT_LENGTH = 3
K = 5
NUM_ACTIONS = 5
NUM_FOLLOWER_ACTIONS = 4
GRID_RESOLUTION = 10  # Controls the coarseness of the uniform grid
T = 1000
base_dir = 'results/'

class OFUL:
    def __init__(self, alpha=1.0, lambda_=1.0):
        self.alpha = alpha                      # parameter to trade-off between exploration & exploitation
        self.lambda_ = lambda_
        self.V = lambda_ * np.eye(K)  # Regularization term
        self.b = np.zeros(K)  # Weighted sum of observed rewards

    def recommend(self, D_t):
        V_inv = np.linalg.inv(self.V)
        theta_hat = V_inv @ self.b

        max_reward = -1000
        best_action = D_t[0]

        # compute UCB on utility for each action in D_t
        for X in D_t:
            r_t = X @ theta_hat + self.alpha * np.sqrt(X @ V_inv @ X)     # optimistic estimate on reward; store in dict to pair with actions
            
            # enumerate list and update max
            if r_t > max_reward:
                max_reward = r_t
                best_action = X

        return best_action

    def observe_utility(self, X_t, r_t):
        self.V += np.outer(X_t, X_t)
        self.b += r_t * X_t

class LogDetFTRL:
    def __init__(self, eta=0.1, alpha=1.0):
        self.eta = eta
        self.alpha = alpha
        self.H = np.eye(K)
        self.sum_gradients = np.zeros((K, K))

    def recommend(self, U_t):
        utilities = []
        for u in U_t:
            u_matrix = np.outer(u, u)
            trace_term = np.trace(self.H @ u_matrix)
            utilities.append((-trace_term, u))
        _, best_u = max(utilities)
        return best_u

    def observe_utility(self, v_t, observed_utility):
        v_matrix = np.outer(v_t, v_t)
        self.sum_gradients += v_matrix * observed_utility
        self.H = np.linalg.inv(self.alpha * np.eye(K) + self.sum_gradients)

class StackelbergGame:
    def __init__(self, bandit_algorithm, leader_utility_function, follower_utility_functions, follower_sequence, context_sequence, action_space):
        self.bandit_algorithm = bandit_algorithm
        self.leader_utility_function = leader_utility_function
        self.follower_utility_functions = follower_utility_functions
        self.follower_sequence = follower_sequence
        self.context_sequence = context_sequence
        self.action_space = action_space        # (finite) set of possible leader mixed strategies

    def play_game(self):
        cumulative_utility = []
        total_utility = 0

        for t in range(T):
            print(f"t = {t}")
            # Step 1: Observe context z_t
            z_t = self.context_sequence[t]
            follower_index = self.follower_sequence[t]

            # Step 2: Define utility set U_t based on context and action space
            U_t = []
            for x in self.action_space:
                u = np.zeros(K)
                for follower_type in range(K):
                    utility = self.leader_utility_function(z_t, x, self.get_follower_best_response(z_t, x, follower_type))
                    # print(f"Utility for follower type {follower_type}: {utility}")
                    u[follower_type] = utility
                U_t.append(u)

            # Step 3: Get utility vector from bandit algorithm
            v_t = self.bandit_algorithm.recommend(U_t)

            # Step 4: Observe utility from realized follower type
            realized_utility = v_t[follower_index]
            total_utility += realized_utility

            # Step 5: Provide utility feedback to the bandit algorithm
            self.bandit_algorithm.observe_utility(v_t, realized_utility)

            # Record cumulative utility
            cumulative_utility.append(total_utility)

        return cumulative_utility

    def get_follower_best_response(self, z_t, x, follower_index):
        # Compute the best response of the follower given context z_t and leader's mixed strategy x
        best_response = -1
        max_utility = float('-inf')
        for action in range(NUM_FOLLOWER_ACTIONS):
            utility = self.follower_utility_functions[follower_index](z_t, x, action)
            if utility > max_utility:
                max_utility = utility
                best_response = action
        return best_response

    def get_action_from_utility(self, z_t, v_t, follower_index):
        # Find the action corresponding to the utility vector
        for x in self.action_space:
            utility_vector = np.array([
                self.leader_utility_function(z_t, x, self.get_follower_best_response(z_t, x, follower_index))
            ])
            if np.array_equal(utility_vector, v_t):
                return x
        raise ValueError("No action matches the given utility vector.")

# Generate random linear mappings for leader and follower utility functions
np.random.seed(42)  # For reproducibility
def generate_leader_utility_function():
    M = np.round(np.random.rand(NUM_ACTIONS, NUM_FOLLOWER_ACTIONS, CONTEXT_LENGTH), 2)  # payoff tensor rounded to 2 decimal places

    def utility_function(z, x, follower_action):
        return sum(
            (z @ M[i, follower_action]) * x[i] for i in range(NUM_ACTIONS)
        )

    return utility_function

def generate_follower_utility_function():
    M = np.round(np.random.rand(NUM_ACTIONS, NUM_FOLLOWER_ACTIONS, CONTEXT_LENGTH), 2)  # payoff tensor rounded to 2 decimal places

    def utility_function(z, x, action):
        return sum(
            (z @ M[i, action]) * x[i] for i in range(NUM_ACTIONS)
        )

    return utility_function

def generate_uniform_grid(grid_resolution, num_actions):
    grid = []
    step_size = 1.0 / grid_resolution
    def recursive_fill(prefix, remaining):
        if remaining == 0:
            if np.isclose(sum(prefix), 1.0):
                grid.append(np.array(prefix))
            return
        for i in range(grid_resolution + 1):
            recursive_fill(prefix + [i * step_size], remaining - 1)
    recursive_fill([], num_actions)
    return grid

def plot_cumulative_utilities(utility_list, label_list, T):
    plt.figure(figsize=(10, 6))
    for idx, utility in enumerate(utility_list):
        plt.plot(range(T), utility, label=label_list[idx])
    plt.xlabel("Time (T)")
    plt.ylabel("Cumulative Expected Utility")
    plt.title("Cumulative Expected Utility vs Time")
    plt.legend()
    plt.grid()
    plt.show()

def generate_game():
    # Define the context space and action space
    context_sequence = [np.round(np.random.rand(CONTEXT_LENGTH), 2) for _ in range(T)]      # random sqeuence of T contexts rounded to 2 decimal places
    action_space = generate_uniform_grid(GRID_RESOLUTION, NUM_ACTIONS)              # uniform grid over the NUM_ACTIONS-dimensional probability simplex 

    # Define leader and follower utility functions
    leader_utility_function = generate_leader_utility_function()
    follower_utility_functions = [
        generate_follower_utility_function() for _ in range(K)
    ]

    # Define the sequence of followers
    follower_sequence = np.random.choice(range(K), size=T)

    game_dict = {}
    game_dict["context_sequence"] = context_sequence
    game_dict["action_space"] = action_space
    game_dict["leader_utility_function"] = leader_utility_function
    game_dict["follower_utility_functions"] = follower_utility_functions
    game_dict["follower_sequence"] = follower_sequence
    
    return game_dict

def run_oful(game_dict):
    context_sequence = game_dict["context_sequence"]
    action_space = game_dict["action_space"]
    leader_utility_function = game_dict["leader_utility_function"]
    follower_utility_functions = game_dict["follower_utility_functions"]
    follower_sequence = game_dict["follower_sequence"]

    # Initialize the OFUL contextual bandit
    oful_bandit = OFUL()

    # Run the game with OFUL
    oful_game = StackelbergGame(
        action_space=action_space,
        bandit_algorithm=oful_bandit,
        leader_utility_function=leader_utility_function,
        follower_utility_functions=follower_utility_functions,
        context_sequence=context_sequence,
        follower_sequence=follower_sequence
    )

    oful_cumulative_utility = oful_game.play_game()
    return oful_cumulative_utility

def run_logdet(game_dict):
    context_sequence = game_dict["context_sequence"]
    action_space = game_dict["action_space"]
    leader_utility_function = game_dict["leader_utility_function"]
    follower_utility_functions = game_dict["follower_utility_functions"]
    follower_sequence = game_dict["follower_sequence"]

    # Initialize the LogDetFTRL bandit
    logdet_bandit = LogDetFTRL()

    # Run the game with LogDetFTRL
    logdet_game = StackelbergGame(
        action_space=action_space,
        bandit_algorithm=logdet_bandit,
        leader_utility_function=leader_utility_function,
        follower_utility_functions=follower_utility_functions,
        context_sequence=context_sequence,
        follower_sequence=follower_sequence
    )

    logdet_cumulative_utility = logdet_game.play_game()
    return logdet_cumulative_utility

fname = base_dir + f"baseline_context={context_str}_follower={follower_str}_n={n}_T={T}_eta={eta}_num_runs={num_runs}_num_leader_actions={num_leader_actions}_num_follower_actions={num_follower_actions}_context_dim_{context_dim}_num_follower_types{num_follower_types}.pkl"
pickle.dump(baseline_run_list, open(fname, 'wb'))

game_dict = generate_game()
oful_cumulative_utility = run_oful(game_dict)
logdet_cumulative_utility = run_logdet(game_dict)

utility_list = [oful_cumulative_utility, logdet_cumulative_utility]
label_list = ["OFUL", "logdet"]

# Plot the cumulative utilities
plot_cumulative_utilities(utility_list, label_list, T)