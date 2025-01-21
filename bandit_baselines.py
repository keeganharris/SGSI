import numpy as np
from itertools import combinations
from csg_icml import generate_games, plot_cumulative_utilities
import pickle, os

class BanditFeedbackBaseline:
    def __init__(self, barycentric_spanner, num_exploration_rounds):
        self.barycentric_spanner = barycentric_spanner
        self.num_exploration_rounds = num_exploration_rounds
        self.estimated_probabilities = None

    def explore(self, action_space, utility_function, context_sequence, follower_sequence):
        """
        Perform the exploration phase to estimate probabilities.
        """
        num_followers = len(self.barycentric_spanner)

        # Initialize probability estimates
        self.estimated_probabilities = np.zeros((num_followers, NUM_ACTIONS))
        counts = np.zeros((num_followers, NUM_ACTIONS))

        for t in range(self.num_exploration_rounds):
            for i, basis_action in enumerate(self.barycentric_spanner):
                context = context_sequence[t]
                follower_type = follower_sequence[t]

                # Play the basis action
                leader_action = action_space[basis_action]
                follower_response = np.argmax([utility_function(context, leader_action, a) for a in range(NUM_ACTIONS)])

                # Update probabilities and counts
                counts[i, follower_response] += 1

        # Compute estimated probabilities for all actions using barycentric spanner
        for i in range(num_followers):
            if np.sum(counts[i]) > 0:
                self.estimated_probabilities[i] = counts[i] / np.sum(counts[i])
            else:
                self.estimated_probabilities[i] = np.zeros(NUM_ACTIONS)  # Handle division by zero

        # Interpolate probabilities for actions not in the spanner
        self.estimated_probabilities_all = np.zeros((num_followers, NUM_ACTIONS))
        for action in range(NUM_ACTIONS):
            weights = np.linalg.lstsq(self.barycentric_spanner.T, np.eye(NUM_ACTIONS)[:, action], rcond=None)[0]
            for follower in range(num_followers):
                self.estimated_probabilities_all[follower, action] = np.dot(weights, self.estimated_probabilities[follower])

    def recommend(self, context, action_space):
        """
        Recommend an action based on estimated probabilities.
        """
        best_action = None
        max_utility = float('-inf')

        for action in range(NUM_ACTIONS):
            expected_utility = sum(
                self.estimated_probabilities_all[follower][action] for follower in range(len(self.barycentric_spanner))
            )

            if expected_utility > max_utility:
                max_utility = expected_utility
                best_action = action_space[action]

        return best_action

# Integration with `game_dict`
def run_algorithm3(game_dict, num_exploration_rounds):
    context_sequence = game_dict["context_sequence"]
    action_space = game_dict["action_space"]
    leader_utility_function = game_dict["leader_utility_function"]
    follower_sequence = game_dict["follower_sequence"]

    # need to transform action_space to vector_space
    vector_space = []
    for action in action_space:
        vect = []
        for follower_type in range(K):
            if 

    # generate barycentric spanner
    barycentric_spanner = generate_barycentric_spanner(action_space)

    # Initialize the algorithm
    algorithm = BanditFeedbackBaseline(barycentric_spanner, num_exploration_rounds)

    # Exploration phase
    algorithm.explore(action_space, leader_utility_function, context_sequence, follower_sequence)

    # Exploitation phase
    cumulative_utility = []
    total_utility = 0

    for t in range(num_exploration_rounds, T):
        context = context_sequence[t]
        follower_type = follower_sequence[t]

        # Recommend an action
        action = algorithm.recommend(context, action_space)

        # Observe the actual utility
        follower_response = np.argmax(
            [leader_utility_function(context, action, a) for a in range(NUM_ACTIONS)]
        )
        observed_utility = leader_utility_function(context, action, follower_response)

        # Update cumulative utility
        total_utility += observed_utility
        cumulative_utility.append(total_utility)

    return cumulative_utility

def generate_barycentric_spanner(vector_space):
    """
    Generate the barycentric spanner for a set of binary vectors (0/1 in each component).
    """
    action_matrix = np.array(vector_space)

    # Check subsets of size K (dimension of spanner)
    for subset in combinations(range(len(action_matrix)), K):
        subset_matrix = action_matrix[list(subset), :]
        if np.linalg.matrix_rank(subset_matrix) == K:  # Found a full-rank basis
            return subset_matrix

    raise ValueError("No full-rank subset of size K found for the given action space.")

# Global variables for easy configurability
CONTEXT_LENGTH = 3
K = 5
NUM_ACTIONS = 5
NUM_FOLLOWER_ACTIONS = 4
GRID_RESOLUTION = 10  # Controls the coarseness of the uniform grid
T = 1000
NUM_RUNS = 1
base_dir = 'results/'
alg3_fname = base_dir + f"num_runs={NUM_RUNS}_T={T}_grid_resolution={GRID_RESOLUTION}_num_follower_actions={NUM_FOLLOWER_ACTIONS}_num_actions={NUM_ACTIONS}_K={K}_context_length={CONTEXT_LENGTH}_alg3.pkl"

if __name__=="__main__":
    print("Generating games")
    game_list = generate_games()

    alg_list = ["alg3"]
    alg_dict = {}
    for alg in alg_list:
        if alg == "alg3":
            alg_dict[alg] = alg3_fname
            if os.path.exists(alg3_fname):
                print("alg3 has already been run.")
            else:
                alg3_utility_list = []
                for idx, game_dict in enumerate(game_list):
                    print(f"alg3 run {idx+1}")
                    alg3_utility_list.append(run_algorithm3(game_dict, num_exploration_rounds=250)) 
                pickle.dump(alg3_utility_list, open(alg3_fname, 'wb'))

    plot_cumulative_utilities(alg_dict)