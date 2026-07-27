# PuppetGait Rebuttal Experiment Priority

This document orders the checkpoints by their value to the three reviewer rebuttals. It is an execution priority, not the row order of PDF Table 2.

## Common protocol

Every ablation checkpoint is trained on CCGR-MINI for 30,000 iterations under its own `OFFICIAL_REBUTTAL_*` save name. Every completed checkpoint should then be evaluated in the following fixed order:

1. **CCGR-MINI in-domain:** use the training YAML with `--phase test`;
2. **CCGR-MINI→CCPG:** report CL/UP/DN/BG and their mean;
3. **CCGR-MINI→SUSTech1K:** report overall and clothing-change Rank-1;
4. **CCGR-MINI→CASIA-B\*:** report NM/BG/CL, their mean, and the view-pair matrix needed for the view-gap analysis.

The checkpoint must remain under `output/CCGR-MINI/...`; a cross-dataset test YAML should change the test data root and `test_dataset_name`, but must preserve the checkpoint-lookup `dataset_name: CCGR-MINI` convention already used by the CCGR→CCPG configs.

For a fair paired claim, all rows being compared must share the same CCGR split, 30k schedule, batch/sampling protocol, LVM, and gait encoder unless the ablated variable explicitly changes one of them.

## Ranked checkpoint training queue

### P0 — Rebuttal-critical checkpoints

These directly establish the paper's framework claim and answer the 4-score reviewer's central concerns. Do not delay them for lower-priority ablations.

| Rank | Checkpoint / setting | Existing train YAML | Why it is this important | Required evaluation |
|---:|---|---|---|---|
| 1 | **MAIN: submitted full PuppetGait** — SAM 3D Body features/geometry, original + A-pose/yaw-0 canonical branches, Sparse TopK4 | `OFFICIAL_REBUTTAL_T2A4_T2B2_T2C3_T2D4_MAIN_CCGR_TRAIN.yaml` | The paper's reference checkpoint and anchor for almost every rebuttal table. It is also required for view-gap, robustness, temporal, and efficiency tests. | All four base evaluations; all MAIN-specific stress tests below |
| 2 | **A3/D1: same SAM 3D Body representation without canonical branch** | `OFFICIAL_REBUTTAL_T2A3_T2D1_SAM3D_NO_LAYOUT_CCGR_TRAIN.yaml` | Together with MAIN, this is the cleanest backbone-matched reverse-direction test of whether canonicalization—not a stronger LVM—improves transfer. | All four base evaluations; the same corruption tests as MAIN |
| 3 | **C1: Direct relocation** | `OFFICIAL_REBUTTAL_T2C1_DIRECT_CCGR_TRAIN.yaml` | Direct will become the simplified default exposition and is the most important comparator for the OT criticism. | All four base evaluations; CASIA-B view-pair/coverage and occlusion tests |
| 4 | **C2: paper-row Dense OT** — no TopK support, eight dense Sinkhorn iterations | `OFFICIAL_REBUTTAL_T2C2_DENSE_OT_CCGR_TRAIN.yaml` | Completes the submitted Direct/Dense/Sparse comparison in the reverse direction and explains why Dense can over-mix. | All four base evaluations; the same view-gap/coverage tests as MAIN and C1 |
| 5 | **Controlled Dense OT with four iterations** | **To create** from C2 with `ot_iters: 4` and a distinct `OFFICIAL_REBUTTAL_*` name | C2 uses eight iterations, whereas MAIN's sparse support is rebalanced for four iterations. This extra control isolates the TopK support effect from the iteration-count difference. | CCGR, CCPG, and view-gap tests first; then the remaining two targets |

The first publishable reverse-direction table is available as soon as ranks 1–4 finish: MAIN vs A3/D1 answers whether canonicalization transfers; MAIN vs C1 vs C2 answers whether Sparse OT adds value beyond Direct and why Dense behaves differently.

### P1 — Strong corroborating evidence

| Rank | Checkpoint / setting | Existing train YAML | Main rebuttal role | Required evaluation |
|---:|---|---|---|---|
| 6 | **A1: DINOv2-S without canonical branch** | `OFFICIAL_REBUTTAL_T2A1_DINOV2S_NO_CANON_CCGR_TRAIN.yaml` | Starts a second backbone-matched with/without-canonicalization pair, addressing the concern that the effect is specific to SAM 3D Body features. | All four base evaluations |
| 7 | **A2: DINOv2-S with canonical branch** | `OFFICIAL_REBUTTAL_T2A2_DINOV2S_CANON_CCGR_TRAIN.yaml` | Completes the DINOv2-S pair. This reproduces the PDF-era dense-thresholded canonical setting rather than silently changing it to TopK4. | All four base evaluations |
| 8 | **B1: ROMP geometry** | `OFFICIAL_REBUTTAL_T2B1_ROMP_CCGR_TRAIN.yaml` | Tests whether the framework survives a different mesh estimator in the reverse transfer direction; important for both mesh-dependence questions. | All four base evaluations after ROMP preprocessing is complete |
| 9 | **BiggerGait DINOv2-G projected-mask baseline** | `OFFICIAL_REBUTTAL_BIGGERGAIT_DINOV2G_PROJMASK_CCGR_TRAIN.yaml` | Provides a high-capacity 2D-LVM comparator for the end-to-end cost/accuracy discussion. It does not replace A3/D1 because it is not backbone matched to PuppetGait. | CCGR and CCPG first; SUSTech1K and CASIA-B\* afterward; full efficiency profile |

### P2 — GAOT sensitivity checkpoints

All variants below copy MAIN and change exactly one parameter. MAIN supplies the default row (`top-k=4`, `δ=0.2`, `ε=0.01`, four sparse rebalance iterations), so the default must not be retrained under another name.

| Rank | Variant to train | Change from MAIN | Proposed save-name suffix | Reviewer question |
|---:|---|---|---|---|
| 10 | Top-k = 1 | `ot_topk_support: 1` | `HPARAM_TOPK1_CCGR` | Requested top-k sensitivity |
| 11 | Top-k = 2 | `ot_topk_support: 2` | `HPARAM_TOPK2_CCGR` | Requested top-k sensitivity |
| 12 | Top-k = 8 | `ot_topk_support: 8` | `HPARAM_TOPK8_CCGR` | Requested top-k sensitivity |
| 13 | Distance threshold δ = 0.1 | `ot_dist_thresh: 0.1` | `HPARAM_DIST01_CCGR` | Locality/radius sensitivity |
| 14 | Distance threshold δ = 0.3 | `ot_dist_thresh: 0.3` | `HPARAM_DIST03_CCGR` | Locality/radius sensitivity |
| 15 | Temperature ε = 0.005 | `ot_temperature: 0.005` | `HPARAM_TEMP0005_CCGR` | Transport sharpness sensitivity |
| 16 | Temperature ε = 0.02 | `ot_temperature: 0.02` | `HPARAM_TEMP002_CCGR` | Transport sharpness sensitivity |
| 17 | Sparse Sinkhorn iterations = 1 | `ot_sparse_rebalance_iters: 1` | `HPARAM_ITER1_CCGR` | Iteration sensitivity |
| 18 | Sparse Sinkhorn iterations = 2 | `ot_sparse_rebalance_iters: 2` | `HPARAM_ITER2_CCGR` | Iteration sensitivity |
| 19 | Sparse Sinkhorn iterations = 8 | `ot_sparse_rebalance_iters: 8` | `HPARAM_ITER8_CCGR` | Iteration sensitivity |

These YAMLs do not yet exist. Their full `trainer_cfg.save_name`, `evaluator_cfg.save_name`, and YAML filename must carry the `OFFICIAL_REBUTTAL_` prefix. Each should receive all four base evaluations. For early rebuttal triage, finish CCGR and CCPG for all variants before launching their SUSTech1K/CASIA-B\* tests.

### P3 — Canonical-template invariance

| Rank | Checkpoint / setting | Existing train YAML | Role |
|---:|---|---|---|
| 20 | **D2: walking pose, yaw 0** | `OFFICIAL_REBUTTAL_T2D2_NO_APOSE_YAW0_CCGR_TRAIN.yaml` | Shows that A-pose reset is not essential once a shared indexed feature frame exists. |
| 21 | **D3: walking pose, yaw 90** | `OFFICIAL_REBUTTAL_T2D3_NO_APOSE_YAW90_CCGR_TRAIN.yaml` | Shows that the exact target yaw is not the conceptual contribution. |

Both still receive all four base evaluations, but they come after the experiments explicitly requested by the low-score reviewers because the submission already contains the forward-direction layout ablation and the reviewer largely accepts its conclusion.

## Test-only experiment queue

The following experiments should reuse trained checkpoints and do not require new recognition-model training unless implementation forces the preprocessing into the training pipeline.

| Priority | Experiment | Checkpoints | Outputs needed |
|---:|---|---|---|
| T1 | CCGR*→CASIA-B\* view-pair and angle-gap analysis | MAIN, C1, C2, A3/D1 | Per-view-pair Rank-1, small/medium/large gap bins, valid canonical coverage, empty-cell ratio |
| T2 | CCGR cross-camera overhead↔frontal analysis | MAIN, C1, C2, A3/D1 | Rank-1 and coverage by camera/view relation |
| T3 | Occlusion stress test | MAIN, C1, A3/D1 | 40/60/80% occlusion; mesh failure, coverage, and Rank-1 |
| T4 | Low-resolution and truncation test | MAIN, A3/D1 | Severity definition, mesh failure, coverage, and Rank-1 |
| T5 | Geometry perturbation test | MAIN, A3/D1 | Mild/moderate/severe perturbation; performance delta and failure threshold |
| T6 | Temporal consistency test | MAIN | Raw per-frame geometry vs moving-average and sequence-average variants; geometry/feature jitter and Rank-1 |
| T7 | Natural clothing/bag subsets | MAIN, A3/D1 | Same-LVM paired results on loose clothing/clothing change and backpack/bag conditions |
| T8 | End-to-end profiling | MAIN, DINOv2-G baseline, existing BigGait/BiggerGait checkpoints | Frozen/trainable/total params, FLOPs, peak memory, runtime/frame, runtime/sequence under one protocol |

## Immediate scheduling recommendation

If four independent training jobs can run concurrently, launch ranks **1–4** first. When any slot frees, use this order:

`5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18 → 19 → 20 → 21`

Begin T1–T8 as soon as their prerequisite checkpoint is available; do not wait for the entire training queue. In particular, MAIN immediately unlocks most test-only experiments, and the MAIN+A3/D1 pair unlocks the most persuasive backbone-matched robustness evidence.

## Current config gaps

- The nine reverse-PDF settings and DINOv2-G training YAML already exist.
- A3/D1 now has paired CCGR→CCPG, CCGR→SUSTech1K, and CCGR→CASIA-B\* test YAMLs. The SUSTech1K and CASIA-B\* test YAMLs for the remaining settings still need to be created and verified.
- The controlled Dense-4-iteration YAML and all sensitivity YAMLs still need to be created.
- No locally present checkpoint matches the `OFFICIAL_REBUTTAL_*` save names recorded in the current matrix; retrieve any running/completed checkpoint from its execution machine before scheduling a duplicate.
- ROMP B1 must wait until the CCGR ROMP geometry preprocessing is complete on the machine assigned to that run.
