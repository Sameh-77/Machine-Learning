#----------------------------------------------------------#
#   Sameh Algharabli --- 2591386 -- CNG 409 Assignment 4   #
#----------------------------------------------------------#

import numpy as np
import math
class HMM:
    def __init__(self, A, B, Pi):
        self.A = A
        self.B = B
        self.Pi = Pi


    def forward_log(self, O: list):
        """
        :param O: is the sequence (an array of) discrete (integer) observations, i.e. [0, 2,1 ,3, 4]
        :return: ln P(O|λ) score for the given observation, ln: natural logarithm
        """
        # The number of states in the length of the initial probabilities
        no_of_states = len(self.Pi)
        # Initializing the matrix of alpha variables (forward variable)
        forward_matrix = np.zeros(shape=(len(O), no_of_states), dtype=np.float)
        c_list = [] # This list is used for the normalization

        # An outer loop to go through the "times", which is observations length
        for t in range(len(O)):
            # a loop to go through each state
            for s in range(no_of_states):
                # at time 0, initialize it with the initial probability
                if t == 0:
                    forward_matrix[t][s] = self.Pi[s] * self.B[s][O[t]]
                else: # when t is not the first one
                    current_probability = 0
                    # This is a loop to find the sum of all alphas multiplied with A
                    # we need it to find alpha(t+1), page 2 in the pdf
                    for c in range(no_of_states):
                        current_probability += forward_matrix[t - 1][c] * self.A[c][s]

                    # now after finding the sum of previous alphas multiplied with A, we can find alpha(t+1)
                    forward_matrix[t][s] = current_probability * self.B[s][O[t]]


            # Here one t is done #
            # So I calculate the sum of probabilites at time t
            time_t_probability = np.sum(forward_matrix[t])

            # Normalization #
            c = 1/time_t_probability # c calculation
            c_list.append(c) # appending c to the list of c's

            # This loop updates all the alphas at time t according to the formula in the pdf page 3
            # formulat --> alpha(normalized) = alpha * c
            for p in range(no_of_states):
                forward_matrix[t][p] = forward_matrix[t][p] * c

        # Here I'm calculating the ln probability = - sum(log(c)) for all c's
        result = -sum([math.log(c) for c in c_list])
        return result

    #====================================================================#

    def viterbi_log(self, O: list):
        """
        :param O: is an array of discrete (integer) observations, i.e. [0, 2,1 ,3, 4]
        :return: the tuple (Q*, ln P(Q*|O,λ)), Q* is the most probable state sequence for the given O
        """
        # The number of states in the length of the initial probabilities
        no_of_states = len(self.Pi)
        probability = 0

        # Initializing the matrix of alpha variables (transision matrix)
        alpha_matrix = np.zeros(shape=(len(O), no_of_states), dtype=np.float)

        # Initializing the state sequence
        state_sequence = np.zeros(shape=(len(O)), dtype=np.int)

        # An outer loop to go through the "times", which is observations length
        for t in range(len(O)):
            # A loop to go through each state
            for s in range(no_of_states):
                # at time 0, initialize it with the initial probability according to initial state
                if t == 0:
                    # I apply to the normalization here, i take the log of the result
                    alpha_matrix[t][s] = math.log(self.Pi[s] * self.B[s][O[t]])
                else: # if the time is not 0, we calculate alpha t+1
                    # getting the state where probabilty is max at t (current t -> (t+1))
                    prev_t_max_state = np.argmax(alpha_matrix[t - 1])

                    # calculating alpha t+1, with applying the normalization
                    alpha_matrix[t][s] = alpha_matrix[t - 1][prev_t_max_state] + math.log(self.A[prev_t_max_state][s] * self.B[s][O[t]])

            # Here, one t is done #
            # find the states that has max probabilty for each observation, at time t
            state_max = np.argmax(alpha_matrix[t])
            state_sequence[t] = state_max

            # Find the max probability for each t, the one for the last observation is our output
            probability = max(alpha_matrix[t])

        return probability, list(state_sequence)