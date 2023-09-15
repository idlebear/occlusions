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

echo "1"
python run_gym.py --prefix noforget-oneside --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --seed 6893 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py --prefix noforget-oneside --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --seed 42 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py --prefix noforget-oneside --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --seed 520 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py --prefix noforget-oneside --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --seed 97 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py --prefix noforget-oneside --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --seed 29348 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
wait
echo "2"
python run_gym.py --prefix noforget-oneside --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --seed 935567 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py --prefix noforget-oneside --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --seed 7 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py --prefix noforget-oneside --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --seed 4913 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py --prefix noforget-oneside --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --seed 84832 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py --prefix noforget-oneside --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --seed 5566 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
wait

# # while (( "$#" )); do
#     # l=$1
#     for s in ${seeds[*]}; do
#         echo ...seed $s
#         python run_gym.py --prefix single-oneside --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Nominal --seed $s --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
#         python run_gym.py --prefix single-oneside --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost None --seed $s --simulation-steps $simulation_steps --runs $runs > /dev/null 2>&1 &
#         python run_gym.py --prefix single-oneside --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --seed $s --simulation-steps $simulation_steps --runs $runs > /dev/null 2>&1 &
#         python run_gym.py --prefix single-oneside --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --seed $s --simulation-steps $simulation_steps --runs $runs > /dev/null 2>&1 &
#         wait
#     done
#     # shift
# # done
