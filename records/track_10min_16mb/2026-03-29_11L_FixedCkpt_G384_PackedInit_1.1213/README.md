## Leaderboard-track submission: 11L FixedCkpt + G384 Packed Init (val_bpb: 1.12134430)

**val_bpb = 1.12134430** (sliding-window exact, stride=64) | **15.97 MB** counted artifact | Modal `8xH100 SXM`, eval-only export path

This folder packages the best current non-TTT `g384` fixed-checkpoint line into a self-contained record-style submission folder. It is **not** a new SOTA claim against merged PR `#549` (`1.1194`), but it is a leaderboard-track submission that would slot behind the current leader if upstream accepts the packaging.

The important packaging detail is that this folder reproduces from a bundled compressed init artifact plus a bundled prepacked sidecar:

- `init_model.ptz`: fixed-checkpoint base artifact
- `prepacked_sbev.bin`: bundled `g384` export sidecar
- `train_gpt.py`: folder-local script with relative defaults baked in

The exact bundled reproduction on Modal `8xH100 SXM` scored:

| Metric | Value |
|---|---:|
| Step-0 / bundled-init diagnostic `val_bpb` | `1.1451` |
| Int6 roundtrip exact `val_bpb` | `1.14511571` |
| **Int6 sliding-window exact `val_bpb`** | **`1.12134430`** |
| Sliding-window exact `val_loss` | `1.89333786` |
| `bytes_model_int6_zstd` | `15,806,941` |
| `bytes_code` (this folder's `train_gpt.py`) | `158,704` |
| **`bytes_total`** | **`15,965,645`** |

### What changed relative to the raw best check

The slightly better raw-checkpoint validation on Modal scored `1.12134338`, but it depended on a kept raw `final_model.pt` that is too large to ship cleanly in a record folder. This folder instead uses the bundled `.ptz` init artifact, which reproduced within `+0.00000092` BPB of the raw result while remaining under the `16,000,000` byte cap.

### Reproduction

Inside this folder, the script defaults are already baked for the packaged path:

```bash
RUN_ID=pr414_fixed_ckpt_b10_k1024_v15424_g384_record \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Expected runtime behavior:

- `ITERATIONS=0`, so there is no new training
- the script loads `./init_model.ptz`
- export uses `./prepacked_sbev.bin`
- the final leaderboard metric is `final_int6_sliding_window_exact`

### Review-sensitive note

This submission is intentionally explicit that the folder bundles a previous compressed init artifact. The counted challenge artifact reported here follows the repo convention of `bytes_model_int6_zstd + bytes_code`, while the folder itself also contains the bundled init artifact needed to reproduce this eval-only export path. If reviewers consider that dependency structure out of scope for leaderboard acceptance, it should not be treated as a record claim.

### Included files

- `train_gpt.py`: folder-local script with packaged defaults
- `init_model.ptz`: bundled fixed-checkpoint init artifact
- `prepacked_sbev.bin`: bundled `g384` export sidecar
- `train.log`: canonical bundled Modal reproduction log
- `train_seed42.log`: same log under explicit seed naming
- `submission.json`: metadata for this packaged submission
