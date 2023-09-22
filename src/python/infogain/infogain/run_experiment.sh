#!/bin/bash

seeds=("6983" "42" "520" "97" "29348" "935567" "7" "4913" "84832" "5566")
runs=1
actors=15
samples=1000
horizon=25
simulation_steps=200
# higgins_weights=( "0.0" "0.01" "0.05" "0.09" "0.1" "0.11" "0.15" "0.2" "0.3" "0.4" "0.5")
higgins_weights=("0.04" )
# our_weights=("0.5" "1.0"  "1.5"  "2.0" "2.5" "3.0" "3.5" "4.0" "4.5" "5.0" "5.5" "6" "6.5" "7"  "7.5" "8" "8.5" "9" "9.5" "10" )
our_weights=("0.005" "0.01" "0.015" )
weight_seeds=("5329" "49" "931843" "2" "833")

prefix_str=""
if [[ $1 ]]
then
    prefix_str="--prefix=$1"
    shift
fi

# for w in ${higgins_weights[*]}; do
#     echo ...weight: $w
#     for s in ${weight_seeds[*]}; do
#         echo ...seed $s
#         python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $w --seed $s --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
#     done
#     wait
# done

# for w in ${our_weights[*]}; do
#     echo ...weight: $w
#     for s in ${weight_seeds[*]}; do
#         echo ...seed $s
#         python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --visibility-weight $w --seed $s --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
#     done
#     wait
# done

echo "1"
anderson_weight=0.65
python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Anderson --visibility-weight $anderson_weight --seed 6893 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Anderson --visibility-weight $anderson_weight --seed 42 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Anderson --visibility-weight $anderson_weight --seed 520 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Anderson --visibility-weight $anderson_weight --seed 97 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Anderson --visibility-weight $anderson_weight --seed 29348 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
wait
echo "2"
python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Anderson --visibility-weight $anderson_weight --seed 935567 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Anderson --visibility-weight $anderson_weight --seed 7 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Anderson --visibility-weight $anderson_weight --seed 4913 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Anderson --visibility-weight $anderson_weight --seed 84832 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Anderson --visibility-weight $anderson_weight --seed 5566 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
wait

# echo "1"
# higgins_weight=0.04
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 6893 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 42 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 520 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 97 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 29348 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# wait
# echo "2"
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 935567 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 7 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 4913 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 84832 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 5566 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# wait

# echo "1"
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --visibility-weight $our_weight --seed 6893 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --visibility-weight $our_weight --seed 42 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --visibility-weight $our_weight --seed 520 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --visibility-weight $our_weight --seed 97 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --visibility-weight $our_weight --seed 29348 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# wait
# echo "2"
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --visibility-weight $our_weight --seed 935567 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --visibility-weight $our_weight --seed 7 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --visibility-weight $our_weight --seed 4913 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --visibility-weight $our_weight --seed 84832 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --visibility-weight $our_weight --seed 5566 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# wait

# runs=3
# our_weight=8.5
# higgins_weight=0.04
# # while (( "$#" )); do
#     # l=$1
#     for s in ${seeds[*]}; do
#         echo ...seed $s
#         python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Nominal --seed $s --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
#         python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost None --seed $s --simulation-steps $simulation_steps --runs $runs > /dev/null 2>&1 &
#         python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed $s --simulation-steps $simulation_steps --runs $runs > /dev/null 2>&1 &
#         python run_gym.py $prefix_str --actors $actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --visibility-weight $our_weight --seed $s --simulation-steps $simulation_steps --runs $runs > /dev/null 2>&1 &
#         wait
#     done
#     # shift
# # done
