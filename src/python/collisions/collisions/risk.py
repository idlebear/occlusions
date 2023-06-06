# Calculating the RISK along a trajectory

import argparse
from enum import Enum
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

# import scienceplots
import seaborn as sb

# plt.style.use(['science', 'ieee'])


class AgentMode(Enum):
    RANDOM = 0
    FIXED = 1


def print_matrix(index, A, cols=11, caption=None):
    N, M = A.shape

    M = min(cols, M)

    if type(index) == list or type(index) == np.ndarray:
        if len(index) != N:
            print("Ignoring improper labels!")
        labels = index
    else:
        labels = [index for _ in range(N)]

    print("%%%%%")
    print("% Table Data for matrix")
    print("%")
    print("\\begin{table*}")
    if caption is not None:
        print(f"\\caption{{{caption}}}")
    else:
        print(f"\\caption{{Matrix}}")

    print("\\label{table:task-time-data}")
    print("\\begin{center}")

    column_str = "\\begin{tabular}{@{} l"
    heading_str = " "
    for m in range(cols):
        column_str += " c "
        heading_str += f"& \\multicolumn{{1}}{{c}}{{{m}}} "
    column_str += " @{}}"
    heading_str += " \\\\"

    print(column_str)
    # print('\\toprule')
    print(heading_str)

    print("\\midrule")

    for n in range(N):
        try:
            s = f" {labels[n]} "
        except IndexError:
            s = " "

        for m in range(M):
            s += f"& {A[n,m]:.3} "
        s += "\\\\"
        print(s)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table*}")
    print("%")
    print("%%%%%")


def main(args):
    num_states = 15
    num_predictions = 10

    # a simple, linear example
    current_state = np.zeros((1, num_states))

    future_states = [current_state]

    # initial distribution -- again, simple with the agent only found in one cell
    current_state[0, 5] = 1.0

    if args.mode == "random":
        behaviour_matrix = np.zeros([num_states, num_states])

        # populate the behaviour matrix assuming random, single step behaviour
        for i in range(num_states):
            for j in range(i - 1, i + 2, 2):
                # BUGBUG -- Note that we are preserving the portion that would flow out of the states under consideration
                #           and inflating the collision risk as a result.  Should probably drop out of range values but then
                #           the rows would no longer sum to 1 -- could always add a filler state to make things proper
                # j = np.clip( j, 0, num_states - 1 )
                if j >= 0 and j < num_states:
                    behaviour_matrix[i, j] += 1.0 / 2.0

        behaviour_matrices = [(1, behaviour_matrix)]
    else:
        # The simple case -- there are three behaviours: either move left, stay, or right until
        #  the end of the prediction interval.
        #
        # LEFT
        left_matrix = np.zeros([num_states, num_states])
        for i in range(1, num_states):
            left_matrix[i, i - 1] = 1

        # # CENTRE
        # centre_matrix = np.zeros( [num_states, num_states] )
        # for i in range( num_states ):
        #     centre_matrix[ i, i ] = 1

        # RIGHT
        right_matrix = np.zeros([num_states, num_states])
        for i in range(num_states - 1):
            if i < num_states:
                right_matrix[i, i + 1] = 1
        # print_matrix( 'Right', right_matrix )

        behaviour_matrices = [
            (1 / 2, left_matrix),
            (1 / 2, right_matrix),
        ]

    # then we can calculate the future states by multiplying powers of the behaviour matrix
    # by the initial state
    x = [current_state for _ in range(len(behaviour_matrices))]

    print(f"\nPredicted agent future occupancy, calculated iteratively:")
    print_matrix(0, current_state)
    for i in range(1, num_predictions):
        next_x = np.zeros_like(current_state)
        row_total = 0
        for index, (prob, matrix) in enumerate(behaviour_matrices):
            x[index] = x[index] @ matrix
            next_x += x[index] * prob

            # wipe the entries of the current state where X is (i.e., X_i = 0)
            x[index][0, i - 1] = 0
            x[index][0, i] = 0
            x[index][0, i + 1] = 0

            row_total += np.sum(x[index]) * prob

        if row_total > 0:
            # and normalize
            for index in range(len(behaviour_matrices)):
                x[index] /= row_total

        print_matrix(i, next_x)
        future_states.append(next_x)

    composite_states = np.array(future_states).squeeze()
    print_matrix(0, composite_states)

    print("%%%%%")
    print("% Table Data for matrix")
    print("%")
    print("\\begin{table*}")
    print(f"\\caption{{Collision Probability}}")

    print("\\label{table:task-time-data}")
    print("\\begin{center}")

    cols = 2
    column_str = "\\begin{tabular}{@{} l c c @{}}"
    heading_str = f"State &  $\\mathbb{{P}}(X_i=1)$ &  $\\mathbb{{P}}_{{\\textsc{{collision}}}}$ \\\\"

    print(column_str)
    # print('\\toprule')
    print(heading_str)

    print("\\midrule")

    prob_no_collision = 1
    collision_upper_bound = 0
    for i in range(1, len(future_states)):
        try:
            X_i = (
                future_states[i][0, i - 1]
                + future_states[i][0, i]
                + future_states[i][0, i + 1]
            )
            prob_no_collision *= 1 - X_i
            collision_upper_bound += X_i
            print(f"$X_{i}$ &  {X_i: .4} & {(1 - prob_no_collision):.4} \\\\")
        except IndexError:
            pass

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table*}")
    print("%")
    print("%%%%%")


def run_trials(args):
    gen = np.random.default_rng(seed=args.seed)

    agent_behaviours = [-1, 1]
    agent_mode = AgentMode.FIXED
    if args.mode == "random":
        agent_mode = AgentMode.RANDOM
    agent_start = 5

    self_start = 0

    result_sample_rate = 10
    results = np.zeros((args.trials,))
    result_summary = []
    result_index = []

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

            if i and not (i % result_sample_rate):
                result_summary.append(np.sum(results) / (i + 1))
                result_index.append(i + 1)

    collision_count = np.sum(results)
    print(
        f"{collision_count} collisions in {args.trials} trials: {collision_count/args.trials} probability of collision"
    )

    ax, fig = plt.subplots()
    plt.plot(result_index, result_summary)
    plt.show()


def generate_observability(generator, steps, p=0.5):
    return generator.choice([True, False], size=steps, replace=True, p=[p, 1.0 - p])


def predict_modal_probability(args):
    # (re)set the random generator
    gen = np.random.default_rng(seed=args.seed)

    collected_data = []

    for observability in np.arange(0.0, 1.01, 0.1):
        collected_collision_vars = []
        collected_collision_predictions = []
        for trial in range(args.trials):
            # find the observability for this trial
            observable = generate_observability(
                generator=gen, steps=args.steps, p=observability
            )

            agent_behaviours = [-1, 1]  # really just left/right...
            agent_mode = AgentMode.FIXED
            num_states = args.steps + 5  # some overflow space if necessary...

            agent_start = 5
            initial_agent_behaviour = 0  # gen.choice(range(len(agent_behaviours)))

            av_belief = np.zeros((len(agent_behaviours), args.steps))

            # a simple, linear example
            current_state = np.zeros((len(agent_behaviours), num_states))

            future_states = []

            # construct the behaviour matrices based on the assigned behaviours
            behaviour_kernals = [
                np.array([1, 0, 0]),  # move left
                np.array([0, 0, 1]),  # move right
                np.array(
                    [
                        1.0 / 3.0,
                        1.0 / 6.0,
                        0,
                    ]
                ),  # unknown left
                np.array(
                    [
                        0,
                        1.0 / 6.0,
                        1.0 / 3.0,
                    ]
                ),  # unknown right
            ]

            if observable[0]:
                # initial distribution -- again, simple with the agent only found in one cell
                current_state[initial_agent_behaviour, agent_start] = 1.0
            else:
                # only have an approximate position, but unable to observe the direction of travel
                current_state[:, agent_start] = 1.0 / len(agent_behaviours)

            future_states.append(current_state)

            for i in range(args.steps):
                # calculate the current belief
                if observable[i]:
                    # we can make an observation -- update the belief based on measured direction
                    next_state = np.array(
                        [
                            np.convolve(
                                current_state[0], behaviour_kernals[0], mode="same"
                            )
                            * args.consistency
                            + np.convolve(
                                current_state[1], behaviour_kernals[0], mode="same"
                            )
                            * (1 - args.consistency),
                            np.convolve(
                                current_state[1], behaviour_kernals[1], mode="same"
                            )
                            * args.consistency
                            + np.convolve(
                                current_state[0], behaviour_kernals[1], mode="same"
                            )
                            * (1 - args.consistency),
                        ]
                    )
                else:
                    # no observation is possible, just a general blur
                    next_state = np.array(
                        [
                            np.convolve(
                                current_state[0], behaviour_kernals[2], mode="same"
                            )
                            + np.convolve(
                                current_state[1], behaviour_kernals[2], mode="same"
                            ),
                            np.convolve(
                                current_state[0], behaviour_kernals[3], mode="same"
                            )
                            + np.convolve(
                                current_state[1], behaviour_kernals[3], mode="same"
                            ),
                        ]
                    )

                # print(next_state)
                # print(f'total probability: {np.sum(next_state)}  (should be 1!)')

                future_states.append(next_state)
                current_state = next_state

                # clear the current footprint of the AV (assuming no collision)
                current_state[:, i - 1 : i + 2] = 0

                # and normalize
                total_probability = np.sum(current_state)
                if total_probability:
                    current_state /= total_probability

            # summarize occupancy (ignore behaviour since it doesn't matter at this point)
            total_occupancy = []
            for state in future_states:
                total_occupancy.append(np.sum(state, axis=0))
            total_occupancy = np.array(total_occupancy).squeeze()

            prob_no_collision = 1
            collision_upper_bound = 0

            for i in range(1, len(future_states)):
                try:
                    X_i = np.sum(future_states[i][:, i - 1 : i + 2])
                    prob_no_collision *= 1 - X_i

                    entry = {
                        "observability": observability,
                        "trial": trial,
                        "X_i": i,
                        "occupancy": X_i,
                        "collision_prob": 1 - prob_no_collision,
                    }
                    collected_data.append(entry)
                except IndexError:
                    pass

            prob_no_collision = 1
            collision_upper_bound = 0
            for i in range(1, len(future_states)):
                try:
                    X_i = np.sum(future_states[i][:, i - 1 : i + 2])
                    prob_no_collision *= 1 - X_i
                    collision_upper_bound += X_i
                    print(f"$X_{i}$ &  {X_i: .4} & {(1 - prob_no_collision):.4} \\\\")
                except IndexError:
                    pass

    df = pd.DataFrame(collected_data)

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)

    sb.lineplot(x="X_i", y="collision_prob", hue="observability", data=df)

    plt.show()

    # # print the results
    # print_matrix(observable, total_occupancy)

    # print('%%%%%')
    # print('% Table Data for matrix')
    # print('%')
    # print('\\begin{table*}')
    # print(f'\\caption{{Collision Probability}}')

    # print('\\label{table:task-time-data}')
    # print('\\begin{center}')

    # cols = 2
    # column_str = '\\begin{tabular}{@{} l c c @{}}'
    # heading_str = f'State &  $\\mathbb{{P}}(X_i=1)$ &  $\\mathbb{{P}}_{{\\textsc{{collision}}}}$ \\\\'

    # print(column_str)
    # # print('\\toprule')
    # print(heading_str)

    # print('\\midrule')

    # prob_no_collision = 1
    # collision_upper_bound = 0
    # for i in range(1, len(future_states)):
    #     try:
    #         X_i = np.sum( future_states[i][:, i-1:i+2])
    #         prob_no_collision *= (1 - X_i)
    #         collision_upper_bound += X_i
    #         print(f'$X_{i}$ &  {X_i: .4} & {(1 - prob_no_collision):.4} \\\\')
    #     except IndexError:
    #         pass

    # print('\\bottomrule')
    # print('\\end{tabular}')
    # print('\\end{center}')
    # print('\\end{table*}')
    # print('%')
    # print('%%%%%')


def run_modal_trial(args):
    gen = np.random.default_rng(seed=args.seed)

    # environment consists of nxm cells, with each cell containing a list of behaviours and their probability of
    # occupation

    agent_behaviours = [-1, 1]

    agent_start = 5
    agent_behaviour_index = 0

    self_start = 0

    result_sample_rate = 10
    results = np.zeros((args.trials,))
    result_summary = np.zeros((args.trials // result_sample_rate,))

    collected_data = []

    for observability in np.arange(0, 1.01, 0.1):
        for trial in range(args.trials):
            # agent starts (or is assumed to start)
            av_agent_pos_belief = [agent_start, agent_start]
            av_agent_behaviour_belief = None
            av_pos = self_start

            # initialize the agent
            agent_behaviour_index = gen.choice(range(len(agent_behaviours)))
            agent_pos = agent_start

            # find the observability for this trial
            observable = generate_observability(
                generator=gen, steps=args.steps, p=observability
            )

            stopped = None
            collided = None
            for step in range(args.steps):
                # make an observation
                if observable[step]:
                    av_agent_behaviour_belief = agent_behaviours[agent_behaviour_index]
                    av_agent_pos_belief = [agent_pos, agent_pos]
                else:
                    # TODO: This is a very crude approximation -- if we can't observe the agent, our
                    #       belief of their behaviour should degrade, not just disappear
                    av_agent_behaviour_belief = None

                # decide to take a step or stop
                if (
                    av_pos + 3 >= av_agent_pos_belief[0]
                ):  # and av_pos <= av_agent_pos_belief[1]:
                    # our next step is fraught with peril
                    stopped = step
                    break

                # step both the agent and the av
                av_pos += 1

                # check whether the agent is going to switch direction
                if gen.random() > args.consistency:
                    agent_behaviour_index = 0 if agent_behaviour_index == 1 else 1

                agent_pos += agent_behaviours[agent_behaviour_index]

                # check for collision
                if abs(agent_pos - av_pos) <= 1:
                    collided = step
                    break

                # update the AV belief
                if av_agent_behaviour_belief is not None:
                    av_agent_pos_belief[0] += av_agent_behaviour_belief
                    av_agent_pos_belief[1] += av_agent_behaviour_belief
                else:
                    av_agent_pos_belief[0] += agent_behaviours[0]
                    av_agent_pos_belief[1] += agent_behaviours[1]

            # post-process and write the results
            unnecessary = False
            if stopped is not None:
                if agent_pos - av_pos > 3:
                    unnecessary = True

            if collided is None and stopped is None:
                # made it to the end
                stopped = args.steps

            entry = {
                "observability": observability,
                "trial": trial,
                "stopped": stopped,
                "collided": collided,
                "unnecessary_stop": unnecessary,
            }

            collected_data.append(entry)

    df = pd.DataFrame(collected_data)

    observability_set = list(set(df["observability"]))
    observability_set.sort()

    print("%%%%%")
    print("% Observability Data")
    print("%")
    print("\\begin{table*}")
    print(f"\\caption{{Collisions and Early Stopping}}")
    print("\\label{table:task-time-data}")
    print("\\begin{center}")

    column_str = "\\begin{tabular}{@{} l c c c @{}}"
    heading_str = f"\%Observable &  Collisions &  Unnecessary Stops & Total Trials \\\\"

    print(column_str)
    print("\\toprule")
    print(heading_str)

    print("\\midrule")

    unnecessary_stops = []
    for obs in observability_set:
        df_slice = df[df["observability"] == obs]
        num_entries = len(df_slice)
        num_collisions = len(df_slice[~df_slice["collided"].isna()])
        num_unnecessary = len(df_slice[df_slice["unnecessary_stop"] == True])
        unnecessary_stops.append(num_unnecessary)

        print(f"{obs:0.4} & {num_collisions} & {num_unnecessary} & {num_entries} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table*}")
    print("%")
    print("%%%%%")

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)

    plt.plot(list(observability_set), unnecessary_stops)

    plt.show()


#
# BAYESIAN updates
#
# Given a sample reading of the position of the agent, we want to predict which behaviour they are using.  Since
# we're assuming the actual position reading is perfect, the similarity to the expected trajectory will be pretty
# high,  But, we can use soft-max to validate...
#

#
# pass in a list of behaviours with action, prior, and measurement probabilty for each
#
# behaviours = [
#   {
#         id: 0,
#         action: -1,
#         prior: 0.5,
#         measure_prob: 0.8
#   },
#   {
#         id: 0,
#         action: 1,
#         prior: 0.5,
#         measure_prob: 0.8
#   },
# ]


def predict_bayesian_probability(args, behaviours):
    # (re)set the random generator
    gen = np.random.default_rng(seed=args.seed)

    collected_data = []

    for observability in np.arange(0.0, 1.01, 0.1):
        # Each AV develops its belief of the agent's behaviour resulting in a tree of possible
        # beliefs, with each node having a branching factor based on the number of behaviours.

        # the probabilities of the previous time step become the priors of the next

        probabilities = [
            [
                [{"prior": p["prior"], "prob": 1.0} for p in behaviours],
            ],
        ]

        # check that the incoming behaviours are a proper distribution
        total_prob = 0
        for b in behaviours:
            total_prob += b["prior"]
        assert total_prob == 1

        # For each time step, update all the behaviours
        for t in range(args.steps):
            # For each prior group

            new_beliefs = []
            for index, prior in enumerate(probabilities[-1]):
                # For each behaviour measurement...
                for behaviour in behaviours:
                    # The Bayesian denominator is the sum of P(Mb|b)P(b) where P(b) is the belief for this behaviour from the
                    # previous round, and P(Mb|b) is the probability of making this measurement given behaviour b.  In one case
                    # it is the probability of a correct measurement, in all other cases, it's the error probability for
                    # behaviour b divided by the number of behaviours - 1 (even distribution)
                    denom = 0
                    for b_i, b in enumerate(behaviours):
                        if b == behaviour:
                            # this is the current behaviour
                            denom += behaviour["measure_prob"] * prior[b_i]["prior"]
                        else:
                            # this is some other behaviour, use the error evenly distributed across
                            # the other behaviours
                            denom += (
                                (1 - behaviour["measure_prob"])
                                * (1.0 / (len(behaviours) - 1.0))
                                * prior[b_i]["prior"]
                            )

                    new_priors = []
                    for b_i, b in enumerate(behaviours):
                        if b == behaviour:
                            # this is the current behaviour
                            prob = behaviour["measure_prob"]
                            numer = behaviour["measure_prob"] * prior[b_i]["prior"]
                        else:
                            # this is some other behaviour, use the error evenly distributed across
                            # the other behaviours
                            prob = (1 - behaviour["measure_prob"]) * (
                                1.0 / (len(behaviours) - 1.0)
                            )
                            numer = prob * prior[b_i]["prior"]
                        new_priors.append(
                            {
                                "prior": numer / denom,
                                "prob": prob * prior[b_i]["prob"],
                            }
                        )

                    new_beliefs.append(new_priors)

            print(new_beliefs)

            # consolidate the new beliefs if there are any duplicates
            consolidated_beliefs = []
            for index, belief in enumerate(new_beliefs):
                consolidated_belief = belief
                dups = []

                for other_index, other_belief in enumerate(new_beliefs[index + 1 :]):
                    same = True
                    for p1, p2 in zip(belief, other_belief):
                        if p1["prior"] != p2["prior"]:
                            same = False
                            break
                    if same:
                        for prior_index, prior in enumerate(consolidated_belief):
                            consolidated_belief[prior_index]["prob"] += other_belief[
                                prior_index
                            ]["prob"]
                        dups.append(other_index + index + 1)
                consolidated_beliefs.append(consolidated_belief)

                for other_index in dups[::-1]:
                    new_beliefs.pop(other_index)

            print(len(consolidated_beliefs), consolidated_beliefs)

            probabilities.append(consolidated_beliefs)

            # Calculate the probability of this behaviour based on the measured behaviour -- this won't necessarily
            # be the same as different behaviours could have different measurement error rates.  In the simple case, where
            # measurement error is a flat (single) percentage, then the probabilities are equal for all the measurement
            # behavours other than the candidate behaviour
            #
            # Which makes this difficult?  What is the proper interpretation here as we only know the probability of
            # correctly measuring the behaviour versus the probability of measuring/identifying any one of the others.  To
            # do this properly we need an error distribution for all the other behaviours.
            #
            # For now, we'll assume that the error case uniformly selects from the other behaviours -- probably not true
            # in practice, but should suffice for a proof of concept.
            #

            # For each belief from the distribution of the previous round


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-c",
        "--consistency",
        default=1,
        type=float,
        help="Tendancy of an agent to keep doing what their already doing [0,1]",
    )
    argparser.add_argument(
        "-m",
        "--mode",
        default="random",
        type=str,
        help="Type of Agent action: fixed | random",
    )
    argparser.add_argument(
        "-o",
        "--observability",
        default=1,
        type=float,
        help="Probabilty of making an observation",
    )
    argparser.add_argument("-s", "--seed", default=None, type=int, help="Random Seed")
    argparser.add_argument(
        "--steps", default=10, type=int, help="Number of steps to take"
    )
    argparser.add_argument("-t", "--trials", default=1000, type=int, help="Random Seed")
    argparser.add_argument(
        "--record-data", action="store_true", help="Record data to disk as frames"
    )
    argparser.add_argument(
        "--show-sim", action="store_true", help="Display the simulation window"
    )

    args = argparser.parse_args()

    # main(args)

    # run_trials(args)

    # predict_modal_probability(args)
    # run_modal_trial(args)

    predict_bayesian_probability(
        args=args,
        behaviours=[
            {"id": 0, "action": -1, "prior": 0.5, "measure_prob": 0.8},
            {"id": 0, "action": 1, "prior": 0.5, "measure_prob": 0.8},
        ],
    )
