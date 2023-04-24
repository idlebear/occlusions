# Calculating the RISK along a trajectory

import argparse
from enum import Enum
import numpy as np


import matplotlib
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee'])


class AgentMode(Enum):
    RANDOM = 0
    FIXED = 1


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

    if args.mode == 'random':
        behaviour_matrix = np.zeros([num_states, num_states])

        # populate the behaviour matrix assuming random, single step behaviour
        for i in range(num_states):
            for j in range(i - 1, i + 2, 2):
                # BUGBUG -- Note that we are preserving the portion that would flow out of the states under consideration
                #           and inflating the collision risk as a result.  Should probably drop out of range values but then
                #           the rows would no longer sum to 1 -- could always add a filler state to make things proper
                # j = np.clip( j, 0, num_states - 1 )
                if j >= 0 and j < num_states:
                    behaviour_matrix[i, j] += 1.0/2.0

        behaviour_matrices = [
            (1, behaviour_matrix)
        ]
    else:
        # The simple case -- there are three behaviours: either move left, stay, or right until
        #  the end of the prediction interval.
        #
        # LEFT
        left_matrix = np.zeros([num_states, num_states])
        for i in range(1, num_states):
            left_matrix[i, i-1] = 1

        # # CENTRE
        # centre_matrix = np.zeros( [num_states, num_states] )
        # for i in range( num_states ):
        #     centre_matrix[ i, i ] = 1

        # RIGHT
        right_matrix = np.zeros([num_states, num_states])
        for i in range(num_states-1):
            if i < num_states:
                right_matrix[i, i+1] = 1
        # print_matrix( 'Right', right_matrix )

        behaviour_matrices = [
            (1/2, left_matrix),
            (1/2, right_matrix),
        ]

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


def run_trials(args):

    gen = np.random.default_rng(seed=args.seed)

    agent_behaviours = [-1, 1]
    agent_mode = AgentMode.FIXED
    if args.mode == 'random':
        agent_mode = AgentMode.RANDOM
    agent_start = 5

    self_start = 0

    result_sample_rate = 10
    results = np.zeros((args.trials,))
    result_summary = np.zeros((args.trials//result_sample_rate,))

    for i in range(args.trials):

        # set initial conditions
        if agent_mode == AgentMode.FIXED:
            agent_action = gen.choice(agent_behaviours)
        agent_pos = agent_start
        self_pos = self_start

        # ten steps along the trajectory
        for p in range(args.steps):

            # move self
            self_pos += 1

            # move agent
            if agent_mode == AgentMode.RANDOM:
                agent_action = gen.choice(agent_behaviours)
            agent_pos += agent_action

            if abs(agent_pos - self_pos) <= 1:
                # collision
                results[i] = 1
                break

        if i and not i % result_sample_rate:
            result_summary[i//result_sample_rate] = np.sum(results)/(i+1)

    collision_count = np.sum(results)
    print(f'{collision_count} collisions in {args.trials} trials: {collision_count/args.trials} probability of collision')

    plt.style.use(['science', 'ieee'])

    plt.figure


def generate_observability(generator, steps, p=0.5):
    return generator.choice([True, False], size=steps, replace=True, p=[p, 1.0-p])


def predict_modal_probability(args):
    # (re)set the random generator
    gen = np.random.default_rng(seed=args.seed)

    # find the observability for this trial
    observable = generate_observability(generator=gen, steps=args.steps, p=args.observability)

    agent_behaviours = [-1, 1]  # really just left/right...
    agent_mode = AgentMode.FIXED
    num_states = args.steps + 5  # some overflow space if necessary...

    agent_start = 5
    initial_agent_behaviour = gen.choice(range(len(agent_behaviours)))

    av_belief = np.zeros((len(agent_behaviours), args.steps))

    # a simple, linear example
    current_state = np.zeros((len(agent_behaviours), num_states))

    future_states = []

    # construct the behaviour matrices based on the assigned behaviours
    behaviour_kernals = [
        np.array([1, 0, 0]),          # move left
        np.array([0, 0, 1]),          # move right
        np.array([1.0/3.0, 1.0/3.0, 1.0/3.0,]),      # unknown
    ]

    if observable[0]:
        # initial distribution -- again, simple with the agent only found in one cell
        current_state[initial_agent_behaviour, agent_start] = 1.0
    else:
        # only have an approximate position, but unable to observe the direction of travel
        current_state[:, agent_start] = 1.0/len(agent_behaviours)

    for i in range(args.steps):
        # calculate the current belief
        if observable[i]:
            # we can make an observation -- update the belief based on measured direction
            next_state = np.array([np.convolve(current_state[0], behaviour_kernals[0], mode='same')*args.consistency +
                                   np.convolve(current_state[1], behaviour_kernals[0], mode='same')*(1-args.consistency),
                                   np.convolve(current_state[1], behaviour_kernals[1], mode='same')*args.consistency +
                                   np.convolve(current_state[0], behaviour_kernals[1], mode='same')*(1-args.consistency)
                                   ])
        else:
            # no observation is possible, just a general blur
            next_state = np.array([np.convolve(current_state[0], behaviour_kernals[2], mode='same'),
                                   np.convolve(current_state[1], behaviour_kernals[2], mode='same')
                                   ])

        print(next_state)
        print(f'total probability: {np.sum(next_state)}  (should be 1!)')

        future_states.append(next_state)
        current_state = next_state

    # predict occupation
    total_occupancy = []
    for state in future_states:
        total_occupancy.append(np.sum(state, axis=0))
    total_occupancy = np.array(total_occupancy).squeeze()

    # print the results
    print_matrix('pred', total_occupancy)


def run_modal_trial(args):

    gen = np.random.default_rng(seed=args.seed)

    # environment consists of nxm cells, with each cell containing a list of behaviours and their probability of
    # occupation

    agent_behaviours = [-1, 1]
    agent_mode = AgentMode.FIXED
    if args.mode == 'random':
        agent_mode = AgentMode.RANDOM
    agent_start = 5

    self_start = 0

    result_sample_rate = 10
    results = np.zeros((args.trials,))
    result_summary = np.zeros((args.trials//result_sample_rate,))

    for i in range(args.trials):
        pass


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--consistency',
        default=1,
        type=float,
        help='Tendancy of an agent to keep doing what their already doing [0,1]')
    argparser.add_argument(
        '-m', '--mode',
        default='random',
        type=str,
        help='Type of Agent action: fixed | random')
    argparser.add_argument(
        '-o', '--observability',
        default=1,
        type=float,
        help='Probabilty of making an observation')
    argparser.add_argument(
        '-s', '--seed',
        default=None,
        type=int,
        help='Random Seed')
    argparser.add_argument(
        '--steps',
        default=10,
        type=int,
        help='Number of steps to take')
    argparser.add_argument(
        '-t', '--trials',
        default=1000,
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

    # main(args)

    # run_trials(args)

    predict_modal_probability(args)
