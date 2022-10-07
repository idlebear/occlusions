import argparse
from importlib.metadata import distribution
from random import seed
from simulation import Simulation
from config import *
import pygame

from importlib import import_module
from os import path, mkdir
from time import time, sleep
from pickle import load, dump
from math import floor


def simulate(args, delivery_log=None):

    if args.show_sim:
        pygame.init()
        size = (args.width, args.height)
        screen = pygame.display.set_mode(size)
        surface = pygame.Surface(size, pygame.SRCALPHA)
        pygame.display.set_caption('Simulation')

        clock = pygame.time.Clock()
        pygame.font.init()
    else:
        screen = None

    # set the seed
    if args.seed is not None:
        seed(args.seed)
        print("Setting seed to: ", args.seed)
    else:
        seed(time())

    generator_args = GENERATOR_ARGS
    generator_args['seed'] = args.seed
    generator_args['max_time'] = args.max_time

    sim = Simulation(
        policy_name=args.policy,
        policy_args={
            'max_v': 0.75,
            'min_v': 0,
            'max_accel': 0.5,
            'max_brake': 0.5,
        },
        generator_name=args.generator,
        generator_args=generator_args,
        num_actors=args.actors,
        pois_lambda=args.lambd,
        screen=surface if args.show_sim or args.record_data else None,
        max_time=args.max_time,
    )

    if args.seed is not None:
        # Save the tasks (or reload them)
        pass

    # Simulation/Game Loop
    while True:
        if args.show_sim:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

        rval = sim.tick(tick_time=args.tick_time, max_simulation_time=args.max_time)
        if rval == -1:
            break

        if args.show_sim:
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            pygame.display.update()
            clock.tick(1 / args.tick_time * args.simulation_speed)

    if len(sim.serviced_tasks) > 0:
        print("Average service time:", sim._avg_served_time/len(sim.serviced_tasks))
    print("Total serviced:", len(sim.serviced_tasks))
    return sim


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--height',
        default=SCREEN_HEIGHT,
        type=int,
        help='Screen vertical size')
    argparser.add_argument(
        '--width',
        default=SCREEN_WIDTH,
        type=int,
        help='Screen horizontal size')
    argparser.add_argument(
        '--margin',
        default=SCREEN_MARGIN,
        type=int,
        help='Screen horizontal size')
    argparser.add_argument(
        '-s', '--seed',
        default=None,
        type=int,
        help='Random Seed')
    argparser.add_argument(
        '-l', '--lambd',
        default=LAMBDA,
        type=float,
        help='Exponential Spawn rate for Tasks')
    argparser.add_argument(
        '-a', '--actors',
        default=NUM_ACTORS,
        type=int,
        help='Number of actors in the simulation')
    argparser.add_argument(
        '-p', '--policy',
        default=DEFAULT_POLICY_NAME,
        help='Policy to use')
    argparser.add_argument(
        '--simulation-speed',
        default=SIMULATION_SPEED,
        type=float,
        help='Simulator speed')
    argparser.add_argument(
        '-t', '--tick-time',
        default=TICK_TIME,
        type=float,
        help='Length of Simulation Time Step')
    argparser.add_argument(
        '-g', '--generator',
        default=DEFAULT_GENERATOR_NAME,
        help='Random Generator to use')
    argparser.add_argument(
        '--load-tasks',
        action='store_true',
        help='Load the most recent list of tasks')
    argparser.add_argument(
        '--max-time',
        default=None,
        type=float,
        help='Maximum Length of Simulation')
    argparser.add_argument(
        '--record-data',
        action='store_true',
        help='Record data to disk as frames')
    argparser.add_argument(
        '--show-sim',
        action='store_true',
        help='Display the simulation window')

    args = argparser.parse_args()

    simulate(args)
