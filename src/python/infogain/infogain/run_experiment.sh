#!/bin/bash

seeds=("6983" "42" "520" "97" "29348" "935567" "7" "4913" "84832" "5566")
runs=1
actors=15
samples=1000
horizon=25
simulation_steps=200

prefix_str=""
if [[ $1 ]]
then
    prefix_str="--prefix=$1"
    shift
fi


# while (( "$#" )); do
    # l=$1
    for s in ${seeds[*]}; do
        echo ...seed $s
        python run_gym.py --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Nominal --seed $s --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
        python run_gym.py --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost None --seed $s --simulation-steps $simulation_steps --runs $runs > /dev/null 2>&1 &
        python run_gym.py --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --seed $s --simulation-steps $simulation_steps --runs $runs > /dev/null 2>&1 &
        python run_gym.py --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --seed $s --simulation-steps $simulation_steps --runs $runs > /dev/null 2>&1 &
        wait
    done
    # shift
# done
