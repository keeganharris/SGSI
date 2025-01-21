import numpy as np
import matplotlib.pyplot as plt
import pickle, os, math
from itertools import combinations

# Global variables for easy configurability
CONTEXT_LENGTH = 3
K = 5
NUM_ACTIONS = 5
NUM_FOLLOWER_ACTIONS = 4
GRID_RESOLUTION = 10  # Controls the coarseness of the uniform grid
T = 1000
NUM_RUNS = 1
FOLLOWER_CONTEXT = False
base_dir = 'results/'
oful_fname = base_dir + f"follower_context={FOLLOWER_CONTEXT}_num_runs={NUM_RUNS}_T={T}_grid_resolution={GRID_RESOLUTION}_num_follower_actions={NUM_FOLLOWER_ACTIONS}_num_actions={NUM_ACTIONS}_K={K}_context_length={CONTEXT_LENGTH}_oful.pkl"
logdet_fname = base_dir + f"follower_context={FOLLOWER_CONTEXT}_num_runs={NUM_RUNS}_T={T}_grid_resolution={GRID_RESOLUTION}_num_follower_actions={NUM_FOLLOWER_ACTIONS}_num_actions={NUM_ACTIONS}_K={K}_context_length={CONTEXT_LENGTH}_logdet.pkl"
alg3_fname = base_dir + f"follower_context={FOLLOWER_CONTEXT}_num_runs={NUM_RUNS}_T={T}_grid_resolution={GRID_RESOLUTION}_num_follower_actions={NUM_FOLLOWER_ACTIONS}_num_actions={NUM_ACTIONS}_K={K}_context_length={CONTEXT_LENGTH}_alg3.pkl"

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

class Algorithm3:
    # Algorithm 3 in NeurIPS paper
    def __init__(self, barycentric_spanner, num_exploration_rounds, action_space, vector_space):
        self.barycentric_spanner = barycentric_spanner
        self.estimated_probabilities = None
        self.num_exploration_rounds = num_exploration_rounds
        self.action_space = action_space
        self.vector_space = vector_space

    def explore(self, follower_utility_functions, context_sequence, follower_sequence):
        """
        Perform the exploration phase to estimate probabilities.
        """
        
        # Initialize probability estimates
        self.estimated_probabilities_bs = np.zeros((K, NUM_FOLLOWER_ACTIONS))
        counts = np.zeros((K, NUM_FOLLOWER_ACTIONS))

        played_actions = []

        for t in range(self.num_exploration_rounds):
            basis_idx = t % K       # circle through basis vectors
            basis_action = self.barycentric_spanner[basis_idx]
            # for basis_idx, basis_action in enumerate(self.barycentric_spanner):
            follower_idx = follower_sequence[t]
            follower_utility_function = follower_utility_functions[follower_idx]

            # Play the basis action
            leader_action = basis_action
            follower_response = np.argmax([follower_utility_function(leader_action, a) for a in range(NUM_FOLLOWER_ACTIONS)])

            # Update probabilities and counts
            counts[basis_idx, follower_response] += 1
            played_actions.append(basis_action)

        # Compute estimated probabilities for all actions using barycentric spanner
        for basis_idx in range(K):
            if np.sum(counts[basis_idx]) > 0:
                self.estimated_probabilities_bs[basis_idx] = counts[basis_idx] / np.sum(counts[basis_idx])
            else:
                self.estimated_probabilities_bs[basis_idx] = np.zeros(NUM_FOLLOWER_ACTIONS)  # Handle division by zero

        # Interpolate probabilities for actions not in the spanner
        self.estimated_probabilities_all = np.zeros((len(self.action_space), NUM_FOLLOWER_ACTIONS))
        spanner_matrix = np.array(self.barycentric_spanner)  # Convert list of vectors to a NumPy array
        
        for idx, leader_action in enumerate(self.action_space):
            weights = np.linalg.lstsq(spanner_matrix.T, self.vector_space[idx])[0]      # compute the weights for each leader action
            for basis_idx in range(K):
                self.estimated_probabilities_all[idx] += weights[basis_idx]*self.estimated_probabilities_bs[basis_idx]
            # self.estimated_probabilities_all[leader_action] = np.dot(weights, self.estimated_probabilities_bs[:, leader_action]) # use weights & BS to compute prob for each action

        # for action in range(NUM_FOLLOWER_ACTIONS):
        #     print(f"Spanner matrix shape: {spanner_matrix.shape}")
        #     print(f"Target vector shape: {np.eye(NUM_FOLLOWER_ACTIONS)[action].shape}")
        #     weights = np.linalg.lstsq(spanner_matrix.T, np.eye(NUM_FOLLOWER_ACTIONS)[:, action])[0]
        #     for follower in range(num_followers):
        #         self.estimated_probabilities_all[follower, action] = np.dot(weights, self.estimated_probabilities[follower])

        return played_actions

    def recommend(self, U_t):       # act greedily w.r.t. the estimated probabilities
        """
        Recommend an action based on estimated probabilities.
        """
        best_action = None
        max_utility = float('-inf')

        for action_index, utility_vector in enumerate(U_t):

            expected_utility = np.dot(self.estimated_probabilities_all[action_index, :], utility_vector)
            if expected_utility > max_utility:
                max_utility = expected_utility
                best_action = utility_vector

        return best_action

    def observe_utility(self, v_t, observed_utility):
        """
        Update the algorithm based on observed utilities (if needed).
        Currently a placeholder for any adaptive updates.
        """
        pass

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
            if t % 100 == 0:
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
            if FOLLOWER_CONTEXT:
                utility = self.follower_utility_functions[follower_index](z_t, x, action)
            else:
                utility = self.follower_utility_functions[follower_index](x, action)
            if utility > max_utility:
                max_utility = utility
                best_response = action
        return best_response

class StackelbergGameBaseline:
    def __init__(self, bandit_algorithm, leader_utility_function, follower_utility_functions, follower_sequence, context_sequence, action_space, explore_length):
        self.bandit_algorithm = bandit_algorithm
        self.leader_utility_function = leader_utility_function
        self.follower_utility_functions = follower_utility_functions
        self.follower_sequence = follower_sequence
        self.context_sequence = context_sequence
        self.action_space = action_space        # (finite) set of possible leader mixed strategies
        self.explore_length = explore_length

    def play_game(self):
        cumulative_utility = []
        total_utility = 0

        if T > self.explore_length:
            played_actions = self.bandit_algorithm.explore(follower_utility_functions=self.follower_utility_functions, context_sequence=self.context_sequence, follower_sequence=self.follower_sequence)
        
        for t in range(self.explore_length):
            z_t = self.context_sequence[t]
            follower_index = self.follower_sequence[t]
            leader_action = played_actions[t]
            realized_utility = self.leader_utility_function(z_t, leader_action, self.get_follower_best_response(leader_action, follower_index))
            total_utility += realized_utility
            cumulative_utility.append(total_utility)

        for t in range(self.explore_length, T):
            if t % 100 == 0:
                print(f"t = {t}")
            
            # Step 1: Observe context z_t
            z_t = self.context_sequence[t]
            follower_index = self.follower_sequence[t]

            # Step 2: Define utility set U_t based on context and action space
            U_t = []
            for x in self.action_space:
                u = np.zeros(NUM_FOLLOWER_ACTIONS)
                for follower_action in range(NUM_FOLLOWER_ACTIONS):
                    utility = self.leader_utility_function(z_t, x, follower_action)
                    # print(f"Utility for follower type {follower_type}: {utility}")
                    u[follower_action] = utility
                U_t.append(u)

            # Step 3: Get utility vector from bandit algorithm
            v_t = self.bandit_algorithm.recommend(U_t)

            # Step 3.5: invert mapping to see what mixed strategy is played
            for x in self.action_space:
                u = np.zeros(NUM_FOLLOWER_ACTIONS)
                for follower_action in range(NUM_FOLLOWER_ACTIONS):
                    utility = self.leader_utility_function(z_t, x, follower_action)
                    # print(f"Utility for follower type {follower_type}: {utility}")
                    u[follower_action] = utility
                if np.linalg.norm(v_t - u) <= 0.001:
                    x_t = x

            # Step 4: Observe utility from realized follower type
            follower_br = self.get_follower_best_response(x_t, follower_index)
            realized_utility = self.leader_utility_function(z_t, x_t, follower_br)
            total_utility += realized_utility

            # Step 5: Provide utility feedback to the bandit algorithm
            self.bandit_algorithm.observe_utility(v_t, realized_utility)

            # Record cumulative utility
            cumulative_utility.append(total_utility)

        return cumulative_utility

    def get_follower_best_response(self, x, follower_index):
        # Compute the best response of the follower given context z_t and leader's mixed strategy x
        best_response = -1
        max_utility = float('-inf')
        for action in range(NUM_FOLLOWER_ACTIONS):
            utility = self.follower_utility_functions[follower_index](x, action)
            if utility > max_utility:
                max_utility = utility
                best_response = action
        return best_response

# Generate random linear mappings for leader and follower utility functions
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

def generate_follower_utility_function_no_context():
    M = np.round(np.random.rand(NUM_ACTIONS, NUM_FOLLOWER_ACTIONS), 2)  # payoff tensor rounded to 2 decimal places

    def utility_function(x, action):
        return sum(
            (M[i, action]) * x[i] for i in range(NUM_ACTIONS)
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

def generate_games(follower_context=True):
    game_list = []
    for run in range(NUM_RUNS):
        # Define the context space and action space
        context_sequence = [np.round(np.random.rand(CONTEXT_LENGTH), 2) for _ in range(T)]      # random sqeuence of T contexts rounded to 2 decimal places
        action_space = generate_uniform_grid(GRID_RESOLUTION, NUM_ACTIONS)              # uniform grid over the NUM_ACTIONS-dimensional probability simplex 

        # Define leader and follower utility functions
        leader_utility_function = generate_leader_utility_function()
        if follower_context:
            follower_utility_functions = [
                generate_follower_utility_function() for _ in range(K)
            ]
        else:
            follower_utility_functions = [
                generate_follower_utility_function_no_context() for _ in range(K)
            ]
            

        # Define the sequence of followers
        follower_sequence = np.random.choice(range(K), size=T)

        game_dict = {}
        game_dict["context_sequence"] = context_sequence
        game_dict["action_space"] = action_space
        game_dict["leader_utility_function"] = leader_utility_function
        game_dict["follower_utility_functions"] = follower_utility_functions
        game_dict["follower_sequence"] = follower_sequence
        game_list.append(game_dict)

    # pickle.dump(game_list, open(game_fname, 'wb'))
    return game_list

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

def run_algorithm3(game_dict, num_exploration_rounds):
    context_sequence = game_dict["context_sequence"]
    action_space = game_dict["action_space"]
    leader_utility_function = game_dict["leader_utility_function"]
    follower_sequence = game_dict["follower_sequence"]
    follower_utility_functions = game_dict["follower_utility_functions"]

    # Create the Stackelberg game instance
    stackelberg_game_baseline = StackelbergGameBaseline(
        bandit_algorithm=None,
        leader_utility_function=leader_utility_function,
        follower_utility_functions=follower_utility_functions,
        follower_sequence=follower_sequence,
        context_sequence=context_sequence,
        action_space=action_space,
        explore_length=num_exploration_rounds
    )

    # go from action space to vector space
    vector_space = []
    for action in action_space:
        for follower_action in range(NUM_FOLLOWER_ACTIONS):
            vect = np.zeros(K)
            for follower_idx in range(K):
                if stackelberg_game_baseline.get_follower_best_response(action, follower_idx) == follower_action:
                    vect[follower_idx] = 1
                else:
                    vect[follower_idx] = 0
            vector_space.append(vect)

    print("About to create Barycentric spanner")
    # Generate barycentric spanner
    barycentric_spanner = generate_barycentric_spanner(vector_space)
    print("Created Barycentric spanner")

    # Initialize Algorithm3
    bandit_algorithm = Algorithm3(barycentric_spanner, num_exploration_rounds, action_space, vector_space)

    # update stackelberg_game_baseline
    stackelberg_game_baseline.bandit_algorithm = bandit_algorithm

    alg3_cumulative_utility = stackelberg_game_baseline.play_game()
    return alg3_cumulative_utility

def generate_barycentric_spanner(action_space):
    """
    Generate the barycentric spanner for a set of binary vectors (0/1 in each component).
    Return the spanner as a list of vectors.
    """
    action_matrix = np.array(action_space)

    # Perform QR factorization to find a basis
    Q, R = np.linalg.qr(action_matrix)

    # Identify the first K linearly independent rows as the spanner
    independent_indices = np.where(np.abs(R.diagonal()) > 1e-10)[0][:K]
    spanner = action_matrix[independent_indices, :]

    if spanner.shape[0] < K:
        raise ValueError("No full-rank subset of size K found for the given action space.")

    return [list(row) for row in spanner]

def plot_cumulative_utilities(alg_dict):
    plt.figure(figsize=(10, 6))
    for alg in alg_dict.keys():
        # compute mean + std
        alg_utility_list = pickle.load(open(alg_dict[alg],'rb'))
        stacked_utilities = np.vstack(alg_utility_list)
        alg_mean = np.mean(stacked_utilities, axis=0)
        alg_std = np.std(stacked_utilities, axis=0)
        plt.plot(range(T), alg_mean, label=alg)
        plt.fill_between(range(T), alg_mean - alg_std, alg_mean + alg_std, alpha=0.2)
    plt.xlabel("Time (T)")
    plt.ylabel("Cumulative Expected Utility")
    plt.title(f"d={CONTEXT_LENGTH}, K={K}, {NUM_ACTIONS} leader actions, {NUM_FOLLOWER_ACTIONS} follower actions")
    plt.legend()
    plt.grid()
    plt.show()

if __name__=="__main__":
    print("Generating games")
    game_list = generate_games(follower_context=FOLLOWER_CONTEXT)

    # game_list = pickle.load(open(game_fname,'rb'))

    alg_list = ["OFUL", "logdet", "alg3"]
    # alg_list = ["alg3"]
    alg_dict = {}
    for alg in alg_list:

        if alg == "OFUL":
            alg_dict[alg] = oful_fname
            if os.path.exists(oful_fname):
                print("OFUL has already been run.")
            else:
                oful_utility_list = []
                for idx, game_dict in enumerate(game_list):
                    print(f"OFUL run {idx+1}")
                    oful_utility_list.append(run_oful(game_dict)) 
                pickle.dump(oful_utility_list, open(oful_fname, 'wb'))

        elif alg == "logdet":
            alg_dict[alg] = logdet_fname
            if os.path.exists(logdet_fname):
                print("logdet has already been run.")
            else:
                logdet_utility_list = []
                for idx, game_dict in enumerate(game_list):
                    print(f"logdet run {idx+1}")
                    logdet_utility_list.append(run_logdet(game_dict)) 
                pickle.dump(logdet_utility_list, open(logdet_fname, 'wb'))
    
        elif alg == "alg3":
            alg_dict[alg] = alg3_fname
            if os.path.exists(alg3_fname):
                print("alg3 has already been run.")
            else:
                alg3_utility_list = []
                for idx, game_dict in enumerate(game_list):
                    print(f"alg3 run {idx+1}")
                    alg3_utility_list.append(run_algorithm3(game_dict, num_exploration_rounds=25)) 
                pickle.dump(alg3_utility_list, open(alg3_fname, 'wb'))

    plot_cumulative_utilities(alg_dict)