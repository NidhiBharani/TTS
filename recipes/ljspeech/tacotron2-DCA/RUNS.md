# Tacotron 2 DCA / LJSpeech — training run log (April 2022)

Two Colab runs, three days apart. The second **resumed** the first and then **diverged to
NaN**. This file records what happened and why: the cause is an interaction between three
config keys that is not visible from any one of them, and the emitted logs that show it are
not in this repo.

> **These two runs are early attempts, not the end of the project.** Training continued
> through July 2022 — six further runs, including VITS trained to **830,000 steps**
> (`vits_ljspeech-July-13-2022`) and a Tacotron checkpoint at step 154,775
> (`run-June-11-2022`), plus GlowTTS and YourTTS inference samples from October 2022. Those
> runs are archived outside git and are **not** described here. Do not read the April
> failures below as the project's outcome; they are its first week.

> **Dates:** all the work described here is from **April 2022**. If this file carries a later
> commit date, that is an archival commit — see [Reading this repo's history](#reading-this-repos-history).

Both runs used `TTS/bin/train_tts.py` on Google Colab against the
`recipes/ljspeech/tacotron2-DCA` recipe, with dataset and outputs on Drive
(`/content/drive/MyDrive/coqui-TTS/LJSpeech-1.1`). The `57223ef94bb2` / `1d73458229f2`
strings in the event filenames are Colab container hostnames.

---

## Run 1 — `run-April-27-2022_08+31AM-0000000`

Started **2022-04-27 08:31:53 UTC**, stopped at **step 1622, epoch 0** — partway through the
first epoch, consistent with a Colab session timeout rather than an error.

Loss fell steeply (total 17.77 → 1.15; `best_model.pth` records `model_loss 1.7266`) but
**`align_error` never improved: 0.871 → 0.965.** Attention alignment had not begun to form,
so the checkpoint does not synthesize intelligible speech.

## Run 2 — `run-April-28-2022_06+50PM-0000000`

Started **2022-04-28 18:50:46 UTC**. **It resumed from Run 1's `best_model.pth`** — Run 1
stopped at step 1622, Run 2's logging begins at **step 1700**. The two are one continuous
training effort.

It reached **step 9700 (epoch 4 of 5)** and **went NaN at step 5700**. Until then it was
training far better than Run 1: loss 6.54 at step 1700 down to **1.50 at step 5600**, the
best value either run reached. Then every scalar became NaN and stayed NaN for the remaining
4000 steps. `align_error` was still ~0.99 when it died.

---

## Why it diverged

The two configs differ in five keys:

| | Run 1 (Apr 27) | Run 2 (Apr 28) |
|---|---|---|
| `lr` | 0.1 | 0.3 |
| `lr_scheduler_params` | `{"warmup_steps": 4000}` | `{"warmup_steps": 0.1}` |
| `epochs` | 3 | 5 |
| `num_loader_workers` | 4 | 2 |
| `save_best_after` | 10000 | 5 |

Two facts about how those were actually applied are not apparent from the configs, and both
are confirmed against the logged `current_lr`:

**1. The `lr: 0.3` never took effect.** Run 2 restored Run 1's optimizer state along with its
weights, and that state carried Run 1's `base_lr` of **0.1**. The logged LR matches
`base_lr = 0.1` to eight significant figures; `base_lr = 0.3` would have produced 0.0949,
which never appears. A config value that is silently overridden on resume is worth knowing
about before trusting any other `lr` in this repo's history.

**2. `warmup_steps` is applied per *epoch*, not per step**, because both configs set
`scheduler_after_epoch: true`. `NoamLR` (from the `trainer` package, not this repo) computes:

```python
step = max(self.last_epoch, 1)
base_lr * warmup_steps**0.5 * min(step * warmup_steps**-1.5, step**-0.5)
```

With `scheduler_after_epoch: true`, `last_epoch` is an epoch counter. Both runs sat at
`last_epoch <= 4`, so the schedule never advanced meaningfully and the LR was determined
almost entirely by `warmup_steps`:

| | `warmup_steps` | effective LR | outcome |
|---|---|---|---|
| Run 1 | 4000 | **2.5e-05**, constant | stable, far too slow — alignment never formed |
| Run 2 | 0.1 | **0.0316**, decaying to 0.0158 by epoch 4 | trained well, then NaN at step 5700 |

That is a **1265× jump** in learning rate between the two runs. The warmup ramp that
`warmup_steps: 4000` was meant to provide never happened in either run — at this scale the
parameter only ever acted as a constant scale factor.

Note that **`0.1` is the `trainer` library's own default** for `warmup_steps`. Run 2's value
was most likely inherited by dropping `lr_scheduler_params` rather than being chosen; it is
not a typo for `4000`.

### If this is picked up again

- Set `scheduler_after_epoch: false` so `warmup_steps` means what it says, or drop NoamLR for
  a scheduler whose behaviour does not depend on that flag.
- Target an LR between the two extremes — Run 1's 2.5e-05 was too low to form alignment,
  Run 2's 0.0316 was unstable past ~5700 steps.
- On resume, set the optimizer LR explicitly rather than trusting the config.
- Resume from **Run 1's** checkpoint. Run 2 saved nothing usable after the NaN, and neither
  run trained long enough to form attention alignment — which remains the real blocker.
- Better still, start from the later 2022 runs rather than either of these. The June/July
  VITS and Tacotron checkpoints are orders of magnitude further along; see the archive.

---

## Where the files are

**In this repo**, under `recipes/ljspeech/tacotron2-DCA/config_files/`:

- `28Apr_config.json` — config as prepared before Run 2, originally committed in `b97fa900`.
- `run-April-28-2022_06+50PM-0000000/config.json` — config Run 2 actually emitted, committed
  the next morning in `1c9cad74`.

Both were **deleted by accident** in `b9dedd24` ("reduce sample sentences from tacotron
evaluation"), whose intended change was 6 lines in `TTS/tts/configs/tacotron_config.py`; it
removed 365 unrelated lines as collateral and never touched `.gitignore`. They are restored
here byte-for-byte from those original blobs.

**Not in this repo:** the TensorBoard event files for both runs and Run 1's `best_model.pth`
(324M). `.gitignore` excludes `events.out*`, and the checkpoint is a large binary. Both are
archived outside git alongside the generated audio samples. Run 2's log is the more
informative: 81 scalar points, 3 evaluation figures (`prediction`, `ground_truth`,
`alignment`) and 1 evaluation audio clip.

## Reading this repo's history

Development here ran **April–July 2022** on a branch named `nidhi`, later renamed `main`.
Any commit dated 2026 is archival — an upload or a note added long after the fact, not work
done on that date. Trust file mtimes, config contents, and the `date` field inside the
checkpoints over commit dates.
