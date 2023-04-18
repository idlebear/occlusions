# Calculating the RISK along a trajectory

import argparse
import numpy as np


def print_matrix(index, A, cols=11, caption=None):
    N, M = A.shape

    M = min(cols, M)

    print('%%%%%')
    print('% Table Data for matrix')
    print('%')
    print('\\begin{table*}')
    if caption is not None:
        print(f'\\caption{{{caption}}}')
    else:
        print(f'\\caption{{Matrix}}')

    print('\\label{table:task-time-data}')
    print('\\begin{center}')

    column_str = '\\begin{tabular}{@{} l'
    heading_str = ' '
    for m in range(cols):
        column_str += ' c '
        heading_str += f'& \\multicolumn{{1}}{{c}}{{{m}}} '
    column_str += ' @{}}'
    heading_str += ' \\\\'

    print(column_str)
    # print('\\toprule')
    print(heading_str)

    print('\\midrule')

    for n in range(N):
        s = f'{index}'
        for m in range(M):
            s += f'& {A[n,m]:.3} '
        s += "\\\\"
        print(s)

    print('\\bottomrule')
    print('\\end{tabular}')
    print('\\end{center}')
    print('\\end{table*}')
    print('%')
    print('%%%%%')


def main(args):

    num_states = 15
    num_predictions = 10

    # a simple, linear example
    current_state = np.zeros((1, num_states))

    future_states = [current_state]

    # initial distribution -- again, simple with the agent only found in one cell
    current_state[0, 5] = 1.0

    behaviour_matrix = np.zeros([num_states, num_states])

    # populate the behaviour matrix assuming random, single step behaviour
    for i in range(num_states):
        for j in range(i - 1, i + 2):
            # BUGBUG -- Note that we are preserving the portion that would flow out of the states under consideration
            #           and inflating the collision risk as a result.  Should probably drop out of range values but then
            #           the rows would no longer sum to 1 -- could always add a filler state to make things proper
            # j = np.clip( j, 0, num_states - 1 )
            if j >= 0 and j < num_states:
                behaviour_matrix[i, j] += 1/3

    behaviour_matrices = [
        (1, behaviour_matrix)
    ]

    # behaviour_matrices = []

    # The simple case -- there are three behaviours: either move left, stay, or right until
    #  the end of the prediction interval.
    #
    # LEFT
    left_matrix = np.zeros([num_states, num_states])
    for i in range(1, num_states):
        left_matrix[i, i-1] = 1

    # CENTRE
    centre_matrix = np.zeros([num_states, num_states])
    for i in range(num_states):
        centre_matrix[i, i] = 1

    # RIGHT
    right_matrix = np.zeros([num_states, num_states])
    for i in range(num_states-1):
        if i < num_states:
            right_matrix[i, i+1] = 1
    # print_matrix( 'Right', right_matrix )

    # behaviour_matrices = [
    #     (1/3, left_matrix),
    #     (1/3, centre_matrix),
    #     (1/3, right_matrix),
    # ]

    # then we can calculate the future states by multiplying powers of the behaviour matrix
    # by the initial state
    x = [current_state for _ in range(len(behaviour_matrices))]

    print(f'\nPredicted agent future occupancy, calculated iteratively:')
    print_matrix(0, current_state)
    for i in range(1, num_predictions):

        next_x = np.zeros_like(current_state)
        row_total = 0
        for index, (prob, matrix) in enumerate(behaviour_matrices):
            x[index] = x[index] @ matrix
            next_x += x[index] * prob

            # wipe the entries of the current state where X is (i.e., X_i = 0)
            x[index][0, i-1] = 0
            x[index][0, i] = 0
            x[index][0, i+1] = 0

            row_total += np.sum(x[index]) * prob

        if row_total > 0:
            # and normalize
            for index in range(len(behaviour_matrices)):
                x[index] /= row_total

        print_matrix(i, next_x)
        future_states.append(next_x)

    composite_states = np.array(future_states).squeeze()
    print_matrix(0, composite_states)

    # alternatively, we can directly calculate the n^th state
    n = 4
    x = np.zeros_like(current_state)

    for index, (prob, matrix) in enumerate(behaviour_matrices):
        x += current_state @ np.linalg.matrix_power(matrix, n) * prob

    print(f'\nThe calculation has a closed form as well -- here is state {n}:')
    print_matrix(n, x)
    print('\n')

    # now risk is evaluated by summing the collision probability of {i-1, i, i+1} for the ith future state

    print('%%%%%')
    print('% Table Data for matrix')
    print('%')
    print('\\begin{table*}')
    print(f'\\caption{{Collision Probability}}')

    print('\\label{table:task-time-data}')
    print('\\begin{center}')

    cols = 2
    column_str = '\\begin{tabular}{@{} l c c @{}}'
    heading_str = f'State &  $\\mathbb{{P}}(X_i=1)$ &  $\\mathbb{{P}}_{{\\textsc{{collision}}}}$ \\\\'

    print(column_str)
    # print('\\toprule')
    print(heading_str)

    print('\\midrule')

    prob_no_collision = 1
    collision_upper_bound = 0
    for i in range(1, len(future_states)):
        try:
            X_i = future_states[i][0, i-1] + future_states[i][0, i] + future_states[i][0, i+1]
            prob_no_collision *= (1 - X_i)
            collision_upper_bound += X_i
            print(f'$X_{i}$ &  {X_i: .4} & {(1 - prob_no_collision):.4} \\\\')
        except IndexError:
            pass

    print('\\bottomrule')
    print('\\end{tabular}')
    print('\\end{center}')
    print('\\end{table*}')
    print('%')
    print('%%%%%')

    # # inclusion/exclusion
    # rv_singles = 0
    # rv_pairs_high = 0
    # rv_pairs_low = 0
    # rv_triplets_high = 0
    # rv_triplets_low = 0

    # for i in range(1, len(future_states)):
    #     X_i = future_states[i][0, i-1] + future_states[i][0, i] + future_states[i][0, i+1]
    #     X_pairs_high = 0
    #     X_pairs_low = 0
    #     X_triplet_high = 0
    #     X_triplet_low = 0

    #     if i > 1:
    #         X_pairs_high = max(future_states[i-1][0, i-1], future_states[i][0, i-1]) +  \
    #             max(future_states[i-1][0, i], future_states[i][0, i])
    #         X_pairs_low = min(future_states[i-1][0, i-1], future_states[i][0, i-1]) +  \
    #             min(future_states[i-1][0, i], future_states[i][0, i])
    #     if i > 2:
    #         X_triplet_high = max(future_states[i-2][0, i-1], future_states[i-1][0, i-1], future_states[i][0, i-1])
    #         X_triplet_low = min(future_states[i-2][0, i-1], future_states[i-1][0, i-1], future_states[i][0, i-1])

    #         X_pairs_high += max(future_states[i-2][0, i-1], future_states[i][0, i-1])
    #         X_pairs_low += min(future_states[i-2][0, i-1], future_states[i][0, i-1])

    #     rv_singles += X_i
    #     rv_pairs_high += X_pairs_high
    #     rv_pairs_low += X_pairs_low
    #     rv_triplets_high += X_triplet_high
    #     rv_triplets_low += X_triplet_low

    #     print(f'(X_{i}): Upper Bound (high): {rv_singles - rv_pairs_low + rv_triplets_high}, (low): {rv_singles - rv_pairs_low + rv_triplets_low} --- Lower (high): {rv_singles - rv_pairs_low }, (low): {rv_singles - rv_pairs_high}')


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-s', '--seed',
        default=None,
        type=int,
        help='Random Seed')
    argparser.add_argument(
        '--record-data',
        action='store_true',
        help='Record data to disk as frames')
    argparser.add_argument(
        '--show-sim',
        action='store_true',
        help='Display the simulation window')

    args = argparser.parse_args()

    main(args)
