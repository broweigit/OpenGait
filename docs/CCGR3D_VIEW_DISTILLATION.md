# CCGR-3D privileged-view distillation

This pipeline trains a PuppetGait teacher and an RGB-only BiggerGait student on
the same deterministic 115/86 identity split. SAM 3D Body is used only for
offline teacher geometry and mask pseudo-labels. The deployed student consists
of DINOv2-S BiggerGait plus a 0.205M-parameter latent view adapter.

Important protocol facts:

- The five released RGB cameras are not synchronized. Training pairs are full
  sequences with the same identity and condition, never same-index frames.
- The paper specifies a random 115/86 split without publishing the sampled
  identities. `datasets/CCGR3D/CCGR3D.json` publishes seed `20260812` and the
  exact identities used here.
- Following the paper, all `NM1` sequences are gallery and other conditions are
  probes. `evaluate_CCGR3D` additionally reports strict cross-view Rank-1.

Run stages in order:

```bash
bash run_ccgr3d_pipeline.sh protocol
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_ccgr3d_pipeline.sh preprocess
bash run_ccgr3d_pipeline.sh visualize-mask
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_ccgr3d_pipeline.sh mask
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_ccgr3d_pipeline.sh teacher
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_ccgr3d_pipeline.sh student
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_ccgr3d_pipeline.sh test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_ccgr3d_pipeline.sh cache
CUDA_VISIBLE_DEVICES=0 bash run_ccgr3d_pipeline.sh adapter
```

Before full preprocessing, use `--limit 2 --debug-vis` manually to verify the
SAM3D projection on the current machine. Preprocessing stores minimal fp16
geometry, canonical A-pose geometry, and a binary 32x16 projection mask. It
does not save the full SAM3D decoder state.

The adapter is trained entirely from sequence-level feature caches. It never
loads DINOv2 or SAM3D, so adapter architecture/loss experiments are cheap.
The real target-view sequence supplies the target-camera feature. PuppetGait's
canonical branch supplies a relational knowledge-distillation target.
