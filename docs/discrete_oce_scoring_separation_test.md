# Discrete OCE Scoring Separation Test Plan

## Summary

The current full-run OCE scoring comparisons are difficult to interpret because the
final entropy metrics combine two effects:

1. the quality of the local candidate scoring rule at a planning decision, and
2. downstream opportunities created after the selected candidate leaves the
   planning horizon.

This experiment separates those effects by freezing a planning decision, scoring
the same candidate set with every method, and then measuring both local selection
quality and downstream rollout outcomes.

Regret in this plan is measured against an exact OCE reference over the finite
candidate set. It is not a claim about the globally optimal path. The primary
reference method is:

```text
oce-gpu-exact-entropy_plus_information
```

For approximation-specific analysis, each approximate scoring mode is also
compared against the exact OCE method with the same scoring mode.

## Method Suite

The separation test should include the full OCE scoring suite plus the baseline
methods used in the existing experiments.

OCE methods:

```text
oce-gpu-exact-entropy
oce-gpu-exact-oc_entropy
oce-gpu-exact-entropy_plus_information
oce-gpu-exact-oc_entropy_plus_information
oce-gpu-exact-information_only

oce-gpu-approximate-entropy
oce-gpu-approximate-oc_entropy
oce-gpu-approximate-entropy_plus_information
oce-gpu-approximate-oc_entropy_plus_information
oce-gpu-approximate-information_only
```

Baselines:

```text
vis-cpu-none-none
none-cpu-none-none
```

The backend placeholder for the baselines should match the existing experiment
labeling convention. Their discrete OCE method and scoring mode placeholders are
`none`.

## Frozen Decision Cases

Each data point is a frozen decision case sampled from a nominal path. A case
captures the complete state needed to regenerate and score the candidate options
without letting different methods change the candidate set.

Each case should store:

- scenario/config identifier
- random seed
- simulation time, tick, or path station
- ego state
- tracker and HMM state
- static map and dynamic occupancy state
- goal
- nominal path
- fixed candidate set
- candidate set hash

Every method must score the same candidate set for a case. Cases with fewer than
two viable candidates should be skipped or explicitly flagged because they cannot
produce meaningful selection/regret comparisons.

## Experiment Design

### Phase 1: Frozen Candidate Scoring

For each frozen decision case:

1. compute scores for every candidate using every method in the method suite;
2. choose the selected candidate for each method;
3. compute the primary exact-reference selected candidate using
   `oce-gpu-exact-entropy_plus_information`;
4. record agreement, exact-reference rank, top-k agreement, and regret; and
5. for approximate OCE methods, additionally compute paired regret against the
   exact OCE method with the same scoring mode.

The primary exact-reference regret is:

```text
reference_score(method_selected_candidate)
-
reference_score(reference_selected_candidate)
```

where `reference_score` is the score assigned by
`oce-gpu-exact-entropy_plus_information`.

The paired approximation regret for a mode such as `entropy` is:

```text
exact_entropy_score(approximate_entropy_selected_candidate)
-
exact_entropy_score(exact_entropy_selected_candidate)
```

This allows the questions:

- Which method most often selects the same candidate as the chosen exact
  reference?
- How much does approximate OCE differ from exact OCE when both use the same
  scoring objective?
- What is the majority selection for each test point?  Which methods agree most often?

Implemented entrypoint:

```text
./run_discrete_oce_separation_phase1.sh
```

The script runs the existing SDD scenario setup once per seed using the nominal
`none` selector for robot motion. At each route replan it scores the fixed
candidate set with the full method suite and writes:

```text
src/python/pedestrian/pedestrian/experiment_logs/<prefix>_phase1_case_methods.csv
src/python/pedestrian/pedestrian/experiment_logs/<prefix>_phase1_candidates.csv
```

The same logging can be enabled in any `main.py` run with:

```text
--discrete-oce-separation-phase1-output <case-method-output.csv>
--discrete-oce-separation-phase1-candidate-output <candidate-output.csv>
```

If the candidate output path is omitted, it defaults to
`<case-method-output-stem>_candidates.csv`.

Phase 1 processing is integrated into the existing uncertainty plotting script:

```text
python experiments/plot_experiment_uncertainty.py experiment_logs --prefix <prefix>
```

The processor auto-detects `<prefix>_phase1_case_methods.csv` and
`<prefix>_phase1_candidates.csv`. Explicit paths may also be passed with:

```text
--phase1-case-methods <case-method-output.csv>
--phase1-candidates <candidate-output.csv>
```

Phase 1 outputs include:

- `phase1_reference_agreement.{png,pdf}`
- `phase1_top2_agreement.{png,pdf}`
- `phase1_majority_agreement.{png,pdf}`
- `phase1_exact_reference_regret.{png,pdf}`
- `phase1_paired_exact_reference_regret.{png,pdf}`
- `phase1_reference_rank_distribution.{png,pdf}`
- `<prefix>_phase1_summary.csv`
- `<prefix>_phase1_selection_table.tex`
- `<prefix>_phase1_regret_table.tex`

### Phase 2: Candidate-To-Goal Rollout

For each method-selected candidate:

1. force the robot through the selected candidate for the planning horizon;
2. after the horizon, roll out to the goal using a common downstream policy; and
3. record final uncertainty, visibility, path, and failure metrics.

The common downstream policy is the primary rollout mode because it isolates the
effect of the initial candidate choice from the effect of repeatedly applying
different planners later in the trajectory.

Visibility and none baselines should be included in both scoring and rollout
phases.

Implemented entrypoint:

```text
./run_discrete_oce_separation_phase2.sh
```

The script runs each initial selector method from the same scenario setup. The
selector chooses the first route candidate, that candidate is forced for the
planning horizon, and the run then switches to a common downstream policy. The
default common policy is `none`; it can be changed with:

```text
COMMON_METHOD=visibility ./run_discrete_oce_separation_phase2.sh
```

For quick smoke runs, the seed set can be narrowed with:

```text
SEEDS="42" ./run_discrete_oce_separation_phase2.sh
```

If the default `python` is not the project environment, pass the interpreter:

```text
PYTHON=/home/bjgilhul/miniconda3/envs/ppo/bin/python ./run_discrete_oce_separation_phase2.sh
```

The script writes:

```text
src/python/pedestrian/pedestrian/experiment_logs/<prefix>_phase2_rollouts.csv
```

Phase 2 processing is integrated into the existing uncertainty plotting script:

```text
python experiments/plot_experiment_uncertainty.py experiment_logs \
  --prefix <prefix> \
  --phase2-rollouts experiment_logs/<prefix>_phase2_rollouts.csv
```

Phase 2 outputs include:

- `phase2_final_state_entropy.{png,pdf}`
- `phase2_final_class_entropy.{png,pdf}`
- `phase2_true_class_probability.{png,pdf}`
- `phase2_visibility_fraction.{png,pdf}`
- `phase2_distance_traveled.{png,pdf}`
- `phase2_goal_rate.{png,pdf}`
- `<prefix>_phase2_summary.csv`
- `<prefix>_phase2_outcome_table.tex`
- `<prefix>_phase2_completion_table.tex`

The processor also generates paired exact-vs-approximate diagnostics for each
seed and scoring mode:

- `<prefix>_phase2_exact_approx_pairs.csv`
- `<prefix>_phase2_exact_approx_pair_table.tex`
- `phase2_exact_approx_state_entropy_delta.{png,pdf}`
- `phase2_exact_approx_class_entropy_delta.{png,pdf}`
- `phase2_exact_approx_true_probability_delta.{png,pdf}`
- `phase2_exact_approx_visibility_delta.{png,pdf}`

These pair rows compare the approximate selector against the exact selector with
the same scoring mode. If matching Phase 1 case-method rows are available, the
pair rows also include the Phase 1 exact-reference regret and paired exact regret.
When plotting a Phase 2 prefix, the script automatically looks for the
corresponding Phase 1 prefix by replacing `phase2` with `phase1`; explicit Phase 1
CSV paths can still be supplied with `--phase1-case-methods`.

### Phase 3: Closed-Loop Method Rollout

For each method-selected candidate:

1. force the robot through the selected candidate for the planning horizon;
2. after the horizon, continue replanning with the same method that selected the
   candidate; and
3. record the same final uncertainty, visibility, path, and failure metrics as
   Phase 2.

This phase is the end-to-end policy test. It measures whether a method's local
scoring choices and downstream replanning behavior work well together in closed
loop. These results should be reported separately from Phase 2 because they
intentionally include compounding effects from later planning decisions.

Visibility and none baselines should be included in the closed-loop rollout as
full methods, not only as references.

Implemented entrypoint:

```text
./run_discrete_oce_separation_phase3.sh
```

The script runs every selector as a full closed-loop policy: OCE methods continue
to replan with their configured exact/approximate scoring objective, visibility
continues to replan with visibility, and none continues to use the nominal
candidate. This phase is the direct comparison for whether exact+info and
approximate+info improve over visibility and none in the complete policy.

As with the other runners, pass the project interpreter if needed:

```text
PYTHON=/home/bjgilhul/miniconda3/envs/ppo/bin/python ./run_discrete_oce_separation_phase3.sh
```

For quick smoke runs:

```text
SEEDS="42" ONLY_SELECTOR=exact_entropy_plus_information ./run_discrete_oce_separation_phase3.sh
```

The script writes:

```text
src/python/pedestrian/pedestrian/experiment_logs/<prefix>_phase3_rollouts.csv
```

Phase 3 processing is integrated into the existing uncertainty plotting script:

```text
python experiments/plot_experiment_uncertainty.py experiment_logs \
  --prefix <prefix> \
  --phase3-rollouts experiment_logs/<prefix>_phase3_rollouts.csv
```

Phase 3 outputs include:

- `phase3_final_state_entropy.{png,pdf}`
- `phase3_final_class_entropy.{png,pdf}`
- `phase3_true_class_probability.{png,pdf}`
- `phase3_visibility_fraction.{png,pdf}`
- `phase3_distance_traveled.{png,pdf}`
- `phase3_goal_rate.{png,pdf}`
- `<prefix>_phase3_summary.csv`
- `<prefix>_phase3_outcome_table.tex`
- `<prefix>_phase3_completion_table.tex`

## Outputs And Metrics

The main output should be a case-method table. Recommended fields:

```text
case_id
seed
scenario
station
method
backend
discrete_oce_method
scoring_mode
num_candidates
candidate_set_hash
selected_index
reference_selected_index
selection_agreement
reference_rank
top2_agreement
top3_agreement
exact_reference_regret
paired_exact_reference_regret
method_score_selected
reference_score_selected
reference_score_best
rollout_mode
experiment_phase
final_sum_state_entropy
final_sum_class_entropy
final_true_class_probability
visibility_fraction
distance_traveled
time_to_goal
timeout
collision
failure_reason
```

An optional case-candidate table should be written when detailed diagnostics are
needed. Recommended fields:

```text
case_id
candidate_index
candidate_set_hash
candidate_path_length
candidate_endpoint
method
method_score
primary_reference_score
paired_exact_reference_score
primary_reference_rank
```

The analysis should report:

- selection agreement with the primary exact reference
- top-2 and top-3 agreement
- exact-reference regret distributions
- paired exact-vs-approximate regret by scoring mode
- rank histograms for selected candidates
- rollout outcome deltas versus the primary reference-selected candidate
- separate Phase 2 common-policy and Phase 3 closed-loop rollout summaries
- state entropy and class entropy means with confidence intervals
- medians and interquartile ranges for heavy-tailed metrics
- visibility, timeout, collision, and distance summaries

Because the current results show wide confidence intervals, rank/agreement and
median/quantile views should be treated as at least as important as mean
confidence intervals.

## Test Plan

Unit tests:

- Use a synthetic case with known candidate scores to verify selected candidate,
  regret, rank, top-k agreement, and tie handling.
- Verify that paired approximate regret compares each approximate scoring mode
  against the exact method with the same scoring mode.
- Verify that the primary reference remains
  `oce-gpu-exact-entropy_plus_information`.

Smoke tests:

- Run one real scenario and confirm that at least one frozen decision case is
  produced.
- Confirm that every method has a row for every valid case.
- Confirm that all methods in a case share the same candidate count and candidate
  set hash.
- Confirm that visibility and none baselines are present with `none`
  placeholders for discrete OCE method and scoring mode.
- Confirm that Phase 2 writes common-policy rollout rows and Phase 3 writes
  closed-loop method rollout rows with distinct `experiment_phase` or
  `rollout_mode` values.

Regression checks:

- Confirm that exact and approximate OCE labels remain distinct in logs, plots,
  and generated tables.
- Confirm that approximate runs do not overwrite exact runs.
- Confirm that cases with fewer than two candidates are skipped or flagged.
- Confirm that rollout failures are represented explicitly rather than silently
  dropped.

## Assumptions

- Exact OCE is a reference oracle over the generated finite candidate set, not
  the unknown global optimum.
- The primary regret reference is
  `oce-gpu-exact-entropy_plus_information`.
- Paired approximation regret compares each approximate scoring mode to the exact
  OCE method with the same scoring mode.
- Lower entropy and lower regret are better.
- Higher visibility and true-class probability are better.
- The primary rollout mode uses a common downstream policy after the forced
  candidate horizon.
- Closed-loop replanning with each method is Phase 3 / Experiment 3. It is the
  end-to-end policy comparison and should be reported separately from the Phase 2
  separation test.
