"""
HMM (Hidden Markov Model) class for calculating the future states of a Markov chain.

This class provides methods to initialize the model parameters, compute the probability of a sequence of observations,
and predict the most likely sequence of hidden states given a sequence of observations. It supports both the forward
and Viterbi algorithms for these computations.

Attributes:
    transition_matrix (numpy.ndarray): The state transition probability matrix.
    emission_matrix (numpy.ndarray): The observation probability matrix.
    state_distribution (numpy.ndarray): The initial state distribution.

Methods:
    forward_algorithm(observations): Computes the probability of the observation sequence using the forward algorithm.
    viterbi_algorithm(observations): Computes the most likely sequence of hidden states using the Viterbi algorithm.
    sample_next_state(current_state): Samples the next state given the current state.
    fit(observations): Fits the model parameters to the given sequence of observations.
"""

import numpy as np
from typing import List, Dict

try:
    from util.dotdict import DotDict
except ImportError:

    class DotDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc


# Tolerance for numerical stability
tolerance = 1e-10  # Avoid division by zero and log of zero errors


# Define the HMM class
class HMM:
    def __init__(
        self,
        num_states: int,
        num_observations: int,
        num_modes: int,
        transitions: np.array,
        emission_probabilities: np.array,
        distributions: dict,
        precompute_diagnostics: bool = True,
        copy_transitions: bool = True,
    ):
        """
        Initializes an instance of the Hidden Markov Model (HMM).

        Args:
            num_states (int): The number of hidden states in the HMM.
            num_observations (int): The number of possible observations.

        Attributes:
            num_states (int): The number of hidden states in the HMM.
            num_observations (int): The number of possible observations.
            num_modes (int): The number of possible transition matrices, one for each defined mode.
            transitions (np.array): A 3D array of shape (num_modes, num_states, num_states) representing the transition probabilities between states for each mode.
            emission_probabilities (np.array): A 2D array of shape (num_states, num_observations) representing the emission probabilities of observations given states.
            distributions (dict): A dictionary containing the initial state and mode distributions with keys 'state' and 'mode'.
        """

        self.num_states = num_states
        self.num_observations = num_observations
        self.num_modes = num_modes

        assert transitions.shape == (num_modes, num_states, num_states)

        assert emission_probabilities is None or emission_probabilities.shape == (
            num_states,
            num_observations,
        )
        self.transition_matrices = (
            transitions.copy() if copy_transitions else transitions
        )
        if emission_probabilities is None:
            # no probabilities supplied - default to uniform
            self.emission_matrix = (
                np.ones((num_states, num_observations)) / num_observations
            )
        else:
            self.emission_matrix = emission_probabilities.copy()

        self.initial_state_distribution = distributions["state"].copy()
        self.initial_mode_distribution = distributions["mode"].copy()

        # These diagnostics are expensive for large SDD state spaces and are not
        # needed by the discrete OCE tracker.
        if precompute_diagnostics:
            self.l2_differences = self.calculate_normalized_l2_distances()
            self.kl_divergences = self.calculate_normalized_kl_divergence()
        else:
            self.l2_differences = None
            self.kl_divergences = None

        self.reset()

    def copy(self):
        """
        Create a deep copy of the HMM that preserves the current belief state.
        """
        hmm_copy = HMM(
            num_states=self.num_states,
            num_observations=self.num_observations,
            num_modes=self.num_modes,
            transitions=self.transition_matrices,
            emission_probabilities=self.emission_matrix,
            distributions={
                "state": self.initial_state_distribution,
                "mode": self.initial_mode_distribution,
            },
            precompute_diagnostics=self.l2_differences is not None,
            copy_transitions=True,
        )

        # Preserve the current belief state and cached inference data
        hmm_copy.state_distribution = self.state_distribution.copy()
        hmm_copy.mode_distribution = self.mode_distribution.copy()
        hmm_copy.alphas = None if self.alphas is None else self.alphas.copy()
        hmm_copy.transition_cache = {
            k: v.copy() for k, v in self.transition_cache.items()
        }

        return hmm_copy

    def reset(self):
        self.state_distribution = self.initial_state_distribution.copy()
        self.mode_distribution = self.initial_mode_distribution.copy()
        self.alphas = None

        self.transition_cache = {}

    def get_mixed_transition_matrix(self, exp=1):
        mixed_transition_matrix = np.zeros(
            (self.num_states, self.num_states), dtype=np.float64
        )
        for mode in range(self.num_modes):
            mixed_transition_matrix += (
                self.get_transition_matrix(mode=mode, exp=exp)
                * self.mode_distribution[mode]
            )
        return mixed_transition_matrix

    def get_transition_matrix(self, mode, exp=1):
        if exp == 1:
            return self.transition_matrices[mode]
        # try:
        #     transition_matrix = self.transition_cache[(mode, exp)]
        # except KeyError:
        #     transition_matrix = np.linalg.matrix_power(
        #         self.transition_matrices[mode], exp
        #     )
        #     self.transition_cache[(mode, exp)] = transition_matrix
        transition_matrix = np.linalg.matrix_power(self.transition_matrices[mode], exp)
        return transition_matrix

    def calculate_normalized_l2_distances(self):
        """
        Calculate the distances between the transition matrices for each mode.
        This is useful for understanding how similar or different the modes are.
        """
        distances = np.zeros((self.num_states, self.num_states))
        for i in range(self.num_modes):
            for j in range(i + 1, self.num_modes):
                for p in range(self.num_states):
                    # Calculate the distance between transition matrices
                    # using Euclidean distance for each state transition
                    distances[p] += (
                        self.transition_matrices[i, p] - self.transition_matrices[j, p]
                    ) ** 2

        return distances

    def calculate_normalized_kl_divergence(self):
        """
        Calculate the Kullback-Leibler divergence between the transition matrices for each mode.
        This is useful for understanding how different the modes are in terms of their transition probabilities.
        """
        divergences = np.zeros((self.num_states, self.num_states))
        for i in range(self.num_modes):
            for j in range(self.num_modes):
                if i != j:
                    for p in range(self.num_states):
                        # Calculate the KL divergence between transition matrices from state p
                        # using the formula: D_KL(P || Q) = sum(P * log(P / Q))
                        P_i = self.transition_matrices[i, p] + tolerance
                        Q_j = self.transition_matrices[j, p] + tolerance
                        # Only add to divergence if P_i > tolerance to avoid log(0)
                        if np.any(P_i > tolerance):
                            divergences[p] += P_i * np.log(P_i / Q_j)

        return divergences

    def transition(self, state_distribution=None, mode_distribution=None, steps=1):
        if state_distribution is None:
            state_distribution = self.state_distribution
        if mode_distribution is None:
            mode_distribution = self.mode_distribution

        next_state_distribution = np.zeros_like(state_distribution)
        for mode in range(self.num_modes):
            next_state_distribution += (
                state_distribution @ self.get_transition_matrix(mode=mode, exp=steps)
            ) * mode_distribution[mode]

        return next_state_distribution

    def sample_next_state(self, mode, state: int, rng: np.random.Generator) -> int:
        # Sample the next state based on the transition probabilities
        return rng.choice(self.num_states, p=self.transition_matrices[mode, state])

    def forward(
        self, observations: List[int], emission_probabilities: np.ndarray = None
    ) -> float:
        """
        Compute the probability of an observation sequence given a Hidden Markov Model (HMM) using the forward algorithm.

        Parameters:
        mode (int): The mode of the transition matrix to use.
        observations (List[int]): A list of observed values.

        Returns:
        float: The probability of the observation sequence.
        """
        alphas = None
        mode_distribution = self.mode_distribution
        for obs in observations:
            mode_distribution, state_distribution, alphas = self.forward_step(
                obs, mode_distribution, alphas, emission_probabilities
            )

        # Return the probability of the observation sequence
        return sum(alphas.sum(axis=1) * mode_distribution)

    def _predict_without_observation(self, steps):
        if self.alphas is None:
            self.state_distribution = self.transition(steps=steps)
            return

        for mode in range(self.num_modes):
            self.alphas[mode] = self.alphas[mode] @ self.get_transition_matrix(
                mode=mode, exp=steps
            )

        Z = float(self.alphas.sum())
        assert Z > 1e-12, "Alpha totals impossible - prediction mismatch!"
        self.alphas /= Z

        # Keep marginals consistent with the joint belief
        self.mode_distribution = self.alphas.sum(axis=1)
        self.state_distribution = self.alphas.sum(axis=0)

    def forward_step(self, observation=None, steps=1, emission_probabilities=None):
        if observation is None:
            self._predict_without_observation(steps=steps)
            return

        if emission_probabilities is None:
            emission_probabilities = self.emission_matrix

        # update alphas
        next_alphas = np.zeros((self.num_modes, self.num_states))
        if self.alphas is None:
            for mode in range(self.num_modes):
                prediction = self.state_distribution @ self.get_transition_matrix(
                    mode=mode, exp=steps
                )
                next_alphas[mode] = (
                    self.mode_distribution[mode]
                    * emission_probabilities[:, observation]
                    * prediction
                )
        else:
            for mode in range(self.num_modes):
                next_alphas[mode] = emission_probabilities[:, observation] * (
                    self.alphas[mode] @ self.get_transition_matrix(mode=mode, exp=steps)
                )
        global_scaler = float(next_alphas.sum())
        assert global_scaler > 1e-10, "Alpha totals impossible - observation mismatch!"

        # scale all alphas together to prevent future underflow using a
        self.alphas = next_alphas / global_scaler

        # update the mode distribution and state distributions
        self.mode_distribution = self.alphas.sum(axis=1)
        self.state_distribution = self.alphas.sum(axis=0)

    def viterbi_algorithm(self, mode: int, observations: List[int]) -> List[int]:
        """
        Applies the Viterbi algorithm to find the most likely sequence of hidden states given a sequence of observations.

        Parameters:
        mode (int): The mode of the transition matrix to use.
        observations (List[int]): A list of observed values.

        Returns:
        List[int]: The most likely sequence of hidden states.

        Note:
        This function was AI generated.
        """
        # Initialize the Viterbi probabilities and backpointers
        viterbi_prob = np.zeros((len(observations), self.num_states))
        backpointers = np.zeros((len(observations), self.num_states), dtype=int)
        for i in range(self.num_states):
            viterbi_prob[0][i] = (
                self.state_distribution[i] * self.emission_matrix[i][observations[0]]
            )

        # Recursively compute the Viterbi probabilities and backpointers
        for t in range(1, len(observations)):
            for j in range(self.num_states):
                max_prob = 0
                max_index = 0
                for i in range(self.num_states):
                    prob = (
                        viterbi_prob[t - 1][i]
                        * self.transition_matrices[mode, i][j]
                        * self.emission_matrix[j][observations[t]]
                    )
                    if prob > max_prob:
                        max_prob = prob
                        max_index = i
                viterbi_prob[t][j] = max_prob
                backpointers[t][j] = max_index

        # Reconstruct the most likely sequence of hidden states
        best_path = [np.argmax(viterbi_prob[-1])]
        for t in range(len(observations) - 1, 0, -1):
            best_path.append(backpointers[t][best_path[-1]])
        best_path.reverse()

        # Return the most likely sequence of hidden states
        return best_path

    def forecast_forward(
        self,
        mode_distribution,
        state_distribution,
        steps=1,
        alphas=None,
        emission_probabilities=None,
    ):
        """
        Forecasts the forward state and mode distributions for a Hidden Markov Model (HMM).

        Parameters:
        mode_distribution (np.array): The mode distribution.
        state_distribution (np.array): The state distribution.
        steps (int): The number of steps to forecast. Default is 1.
        alphas (np.array): The (num_modes x num_states) matrix of alpha vectors.

        Returns:
        Tuple[np.array, np.array, np.array]: The forecasted mode distribution, state distribution, and alpha vectors.

        """

        if emission_probabilities is None:
            emission_probabilities = self.emission_matrix

        # calculate the state update of the alphas
        # next_state_distribution = np.zeros_like(state_distribution)
        next_alphas_states = np.zeros((self.num_modes, self.num_states))
        if alphas is None:
            if state_distribution is None:
                raise ValueError("Either alphas or state_distribution must be provided")

            # evolve the state distribution according to the transition matrices
            for mode in range(self.num_modes):
                next_alphas_states[mode, ...] = (
                    state_distribution
                    @ self.get_transition_matrix(mode=mode, exp=steps)
                )
        else:
            # evolve the state distribution according to the alphas
            for mode in range(self.num_modes):
                next_alphas_states[mode, ...] = alphas[
                    mode
                ] @ self.get_transition_matrix(mode=mode, exp=steps)

        # broadcast the M x N matrix to num_observations x M x N
        next_alphas = np.zeros((self.num_observations, self.num_modes, self.num_states))
        for observation in range(self.num_observations):
            for mode in range(self.num_modes):
                next_alphas[observation, mode, ...] = (
                    emission_probabilities[:, observation]
                    * next_alphas_states[mode, ...]
                )

        # # calculate the alpha totals for each mode and observation
        alpha_totals = next_alphas.sum(axis=2)

        # calculate the next mode distribution for each observation
        next_mode_distribution = alpha_totals * mode_distribution

        # normalize and remove any NaN values
        with np.errstate(divide="ignore", invalid="ignore"):
            next_mode_distribution /= next_mode_distribution.sum(axis=1).reshape(-1, 1)
        next_mode_distribution[np.isnan(next_mode_distribution)] = 0.0

        zero_modes = next_mode_distribution.sum(axis=1) == 0
        next_mode_distribution[zero_modes] = mode_distribution

        assert np.allclose(next_mode_distribution.sum(axis=1), 1.0)

        # # calculate the next state distribution for each observation - a O x N matrix
        # next_state_distribution = np.zeros((self.num_observations, self.num_states))
        # for mode in range(self.num_modes):
        #     next_state_distribution += (
        #         next_alphas[:, mode, :] / alpha_totals[:, mode].reshape(-1, 1)
        #     ) * next_mode_distribution[:, mode].reshape(-1, 1)

        # normalize each alpha set (by observation) to ensure numerical stability
        for observation in range(self.num_observations):
            scalar = next_alphas[observation].sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                next_alphas[observation] /= scalar
        next_alphas[np.isnan(next_alphas)] = 0.0

        return DotDict(
            {
                "mode": next_mode_distribution,
                # "state": next_state_distribution,
                "alphas": next_alphas,
            }
        )


def predict_distributions_for_all_future_observations_at_k(
    hmm: HMM,
    emission_probabilities_at_k: np.ndarray,  # Shape (num_states, num_possible_Yk_values)
) -> List[Dict[str, any]]:
    """
    Predicts, from t=0, the distributions P(X_k|Y_k,X_0), P(M|Y_k,X_0) and the
    probability P(Y_k|X_0) for all possible observations Y_k at a future step k.

    The process involves:
    1. Initializing an HMM with P(X_0) and P(M_0).
    2. Evolving the HMM predictively for k-1 steps to get P(X_{k-1}|X_0) and P(M|X_0).
    3. Using the HMM's forecast_forward method (which considers one more transition
       and all possible emissions at step k) to get P(M|Y_k,X_0) and related alpha values.
    4. Deriving P(Y_k|X_0) and P(X_k|Y_k,X_0) from these results.

    Args:
        initial_state_belief: The initial belief over states P(X_0).
        initial_mode_belief: The initial belief over modes P(M_0).
        k: The future step number (1-indexed) for which to predict observation outcomes.
        transition_matrices: Transition probabilities P(X_t | X_{t-1}, M).
                             Shape (num_modes, num_states, num_states).
        emission_probabilities_at_k: Emission probabilities P(Y_k | X_k) for step k.
                                     Shape (num_states, num_possible_Yk_values).
                                     num_possible_Yk_values is the number of distinct
                                     observations Y_k can take.

    Returns:
        A list of dictionaries. Each dictionary corresponds to one possible observation y_k
        that can occur at step k and contains:
            'observation_value': The index of the observation y_k.
            'observation_prob': P(Y_k=y_k | X_0), the probability of this observation occurring.
            'state_belief_post_obs': P(X_k | Y_k=y_k, X_0), the posterior state belief.
            'mode_belief_post_obs': P(M | Y_k=y_k, X_0), the posterior mode belief.
    """

    # Use forecast_forward for the transition from X_{k-1} to X_k, considering all possible Y_k.
    # hmm.state_distribution is P(X_{k-1}|X_0)
    # hmm.mode_distribution is P(M|X_0)
    # hmm.emission_matrix is emission_probabilities_at_k
    forecast_results = hmm.forecast_forward(
        mode_distribution=hmm.mode_distribution,
        state_distribution=hmm.state_distribution,
        steps=1,
        emission_probabilities=emission_probabilities_at_k,
    )

    # forecast_results.mode is P(M | Y_k=obs, X_0) of shape (num_possible_yk_values, num_modes)
    mode_belief_post_obs_all_y = forecast_results.mode

    # forecast_results.alphas is P(Y_k=obs, X_k | M, X_0) of shape (num_possible_yk_values, num_modes, num_states)
    # P(Y_k=obs, X_k=x_k | M, X_0) = P(Y_k=obs | X_k=x_k) * P(X_k=x_k | M, X_0)
    # where P(X_k=x_k | M, X_0) was derived from P(X_{k-1}|X_0) and T_M.
    alphas_joint_Yk_Xk_given_M_X0 = forecast_results.alphas

    # Calculate P(Y_k=obs | X_0)
    # P(Y_k=obs | M, X_0) = sum_{x_k} P(Y_k=obs, X_k=x_k | M, X_0)
    P_Yk_given_M_X0 = np.sum(
        alphas_joint_Yk_Xk_given_M_X0, axis=2
    )  # Shape (num_obs, num_modes)
    # P(Y_k=obs | X_0) = sum_M P(Y_k=obs | M, X_0) * P(M | X_0)
    # hmm.mode_distribution is P(M|X_0)
    prob_Yk_given_X0_all_y = np.sum(
        P_Yk_given_M_X0 * hmm.mode_distribution[np.newaxis, :], axis=1
    )  # Shape (num_obs,)

    # # Calculate P(X_k | Y_k=obs, X_0)
    # # P(X_k=x_k, Y_k=obs | X_0) = sum_M P(Y_k=obs, X_k=x_k | M, X_0) * P(M | X_0)
    # P_Xk_Yk_joint_given_X0 = np.sum(
    #     alphas_joint_Yk_Xk_given_M_X0
    #     * hmm.mode_distribution[np.newaxis, :, np.newaxis],
    #     axis=1,
    # )  # Shape (num_obs, num_states)

    # state_belief_post_obs_all_y = np.zeros_like(P_Xk_Yk_joint_given_X0)
    # # P(X_k | Y_k=obs, X_0) = P(X_k, Y_k=obs | X_0) / P(Y_k=obs | X_0)
    # for obs_idx in range(num_possible_yk_values):
    #     if prob_Yk_given_X0_all_y[obs_idx] > 1e-15:  # Avoid division by zero
    #         state_belief_post_obs_all_y[obs_idx, :] = (
    #             P_Xk_Yk_joint_given_X0[obs_idx, :] / prob_Yk_given_X0_all_y[obs_idx]
    #         )
    #     else:
    #         # If P(Y_k=obs|X_0) is zero, this observation is impossible.
    #         # The posterior P(X_k|Y_k=obs,X_0) is ill-defined.
    #         # We can set it to a default, e.g., uniform or the predictive P(X_k|X_0).
    #         # Let's use the predictive belief P(X_k|X_0) for consistency with mode belief handling in HMM.
    #         if not hasattr(hmm, "_cached_pred_Xk_given_X0"):
    #             hmm._cached_pred_Xk_given_X0 = hmm.transition(
    #                 state_distribution=hmm.state_distribution,  # P(X_{k-1}|X_0)
    #                 mode_distribution=hmm.mode_distribution,  # P(M|X_0)
    #                 steps=1,
    #             )  # This is P(X_k|X_0)
    #         state_belief_post_obs_all_y[obs_idx, :] = hmm._cached_pred_Xk_given_X0

    results_for_all_obs = {
        "obs_prob": prob_Yk_given_X0_all_y,
        # "state_belief_post_obs": state_belief_post_obs_all_y,
        "mode_belief_post_obs": mode_belief_post_obs_all_y,
    }
    return results_for_all_obs
