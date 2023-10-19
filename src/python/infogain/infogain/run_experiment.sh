#!/bin/bash

seeds=("6983" "42" "520" "97" "29348" "935567" "7" "4913" "84832" "5566")
weight_seeds=("5329" "49" "931843" "2" "833")

runs=3
actors=1
limit_actors="--limit-actors"
samples=1000
horizon=25
simulation_steps=200
# higgins_weights=( "5" "10" "50" "100" "125" "150" "175" "200" )
# higgins_weights=( "10" "50" "60" "70" "80" "90" "100" "110" "120" "130" "140" "150" )
higgins_weights=( "31" ) # "27.5" "30" "32.5" "35" "37.5" ) # "45" "50" "51" "52" "53" "54" "55" "56" "57" "58" "59" )
# our_weights=( "1000" "2500" "5000"  "7500"  "10000" "11000" "12000" "13000" "14000" "15000" "16000" "18000" "20000" )
our_weights=( "7600" "7800" "8000" "8100" "8200"  "8300"  "8400" "8500"  )
# andersen_weights=( "1" "5"  "8" "9" "9.5" "10" "10.5" "11"  "13" "15" "20" ) #"14" "16" "17" "18" "19" "40" "50" )
# andersen_weights=( "500" "750" "1000"  "1500" "2000" "2500" ) #"14" "16" "17" "18" "19" "40" "50" )
# andersen_weights=( "1000" "2500" "5000"  "7500"  "10000" "11000" "12000" "13000" "14000" "15000" "16000" "18000" "20000" )
# andersen_weights=( "11800" "12000" "12100" "12200" "12300" "12400" "12500" )
andersen_weights=( "13000" "12750"  )

# velocities=("0.0" "1.0"  "2.0" "3.0" "4.0"  "5.0" "6.0" "7.0" "8.0" "9.0" "10.0" "15")
velocity_weights=("500" "1000.0" "1500" "2000" "2500" "3000" "3500" "4000" "4500" "5000" "5500" )
y_weights=( "50" "100" "150" "200" "250" "300" "350" "400" "450" "500" )

prefix_str=""
if [[ $1 ]]
then
    prefix_str="--prefix=$1"
    shift
fi

y_weight=300
velocity_weight=3500  # calibrated with lambda=20000

# for w in ${andersen_weights[*]}; do
#     echo ...weight: $w
#     for s in ${weight_seeds[*]}; do
#         echo ...seed $s
#         python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Andersen --visibility-weight $w --velocity-weight $velocity_weight --y-weight $y_weight --seed $s --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
#     done
#     wait
# done

# for w in ${higgins_weights[*]}; do
#     echo ...weight: $w
#     for s in ${weight_seeds[*]}; do
#         echo ...seed $s
#         python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $w  --velocity-weight $velocity_weight --y-weight $y_weight --seed $s --simulation-steps $simulation_steps --runs $runs > /dev/null 2>&1 &
#     done
#     wait
# done

# for w in ${our_weights[*]}; do
#     echo ...weight: $w
#     for s in ${weight_seeds[*]}; do
#         echo ...seed $s
#         python run_gym.py $prefix_str --actors $actors $limit_actors --limit-actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours --visibility-weight $w  --velocity-weight $velocity_weight --y-weight $y_weight --seed $s --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
#     done
#     wait
# done

# y_weight=300
# for v in ${velocity_weights[*]}; do
#     echo ...velocity: $v
#     for s in ${weight_seeds[*]}; do
#         echo ...seed $s
#         python run_gym.py $prefix_str --actors $actors --limit-actors --method mppi --samples $samples --horizon $horizon --visibility-cost Nominal --velocity-weight $v --y-weight $y_weight --seed $s --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
#     done
#     wait
# done

# for yw in ${y_weights[*]}; do
#     echo ...y-weight: $yw
#     for s in ${weight_seeds[*]}; do
#         echo ...seed $s
#         python run_gym.py $prefix_str --actors $actors --limit-actors --method mppi --samples $samples --horizon $horizon --visibility-cost Nominal --y-weight $yw --seed $s --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
#     done
#     wait
# done


# echo "1"
# andersen_weight=43.0
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Andersen --visibility-weight $andersen_weight --seed 6893 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Andersen --visibility-weight $andersen_weight --seed 42 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Andersen --visibility-weight $andersen_weight --seed 520 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Andersen --visibility-weight $andersen_weight --seed 97 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Andersen --visibility-weight $andersen_weight --seed 29348 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# wait
# echo "2"
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Andersen --visibility-weight $andersen_weight --seed 935567 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Andersen --visibility-weight $andersen_weight --seed 7 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Andersen --visibility-weight $andersen_weight --seed 4913 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Andersen --visibility-weight $andersen_weight --seed 84832 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Andersen --visibility-weight $andersen_weight --seed 5566 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# wait

# echo "1"
# actors=15
# higgins_weight=0.11
# limit_actors="" # "--limit-actors"
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 6893 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 42 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 520 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 97 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 29348 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# wait
# echo "2"
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 935567 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 7 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 4913 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 84832 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
# python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins --visibility-weight $higgins_weight --seed 5566 --simulation-steps $simulation_steps --runs $runs  > /dev/null 2>&1 &
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

runs=5
y_weight=300
velocity_weight=3500
actors=1
limit_actors="--limit-actors"
andersen_weight=13000
our_weight=8400
higgins_weight=31
# while (( "$#" )); do
    # l=$1
    for s in ${seeds[*]}; do
        echo ...seed $s
        python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Nominal                                       --velocity-weight $velocity_weight --y-weight $y_weight --seed $s --simulation-steps $simulation_steps --runs $runs > /dev/null 2>&1 &
        python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost None                                          --velocity-weight $velocity_weight --y-weight $y_weight --seed $s --simulation-steps $simulation_steps --runs $runs > /dev/null 2>&1 &
        python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Andersen --visibility-weight $andersen_weight --velocity-weight $velocity_weight --y-weight $y_weight --seed $s --simulation-steps $simulation_steps --runs $runs > /dev/null 2>&1 &
        python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Higgins  --visibility-weight $higgins_weight  --velocity-weight $velocity_weight --y-weight $y_weight --seed $s --simulation-steps $simulation_steps --runs $runs > /dev/null 2>&1 &
        python run_gym.py $prefix_str --actors $actors $limit_actors --method mppi --samples $samples --horizon $horizon --visibility-cost Ours     --visibility-weight $our_weight      --velocity-weight $velocity_weight --y-weight $y_weight --seed $s --simulation-steps $simulation_steps --runs $runs > /dev/null 2>&1 &
        wait
    done
    # shift
done
