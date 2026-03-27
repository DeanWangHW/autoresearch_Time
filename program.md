# autoresearch_Time

This is an experiment to have the LLM do its own research on PatchTST for ETTh1 forecasting.

The fixed task is:

- dataset: `ETTh1`
- horizon: `96`
- train entrypoint: `train.py`
- training file: `patchtst/dataset/ETTh1_train.csv`
- blind evaluation file: `patchtst/dataset/ETTh1_blind_test.csv`

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar27`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `main`.
3. **Verify the conda environment**: Make sure the local `patchtst` conda environment from `patchtst/environment-macos-arm64.yml` exists and is active. If not, tell the human to rebuild it.
4. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed system check and data split preparation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `program.md` — the experiment protocol you must follow.
5. **Verify data exists**: Run `python prepare.py`. This must succeed and must verify:
   - `patchtst/dataset/ETTh1_train.csv`
   - `patchtst/dataset/ETTh1_blind_test.csv`
6. **Initialize results.tsv**: Create `results.tsv` with just the header row if needed. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time). With the `patchtst` environment active, you launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, patch settings, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed system check and split preparation.
- Modify `patchtst/dataset/ETTh1.csv`.
- Modify `patchtst/dataset/ETTh1_train.csv`.
- Modify `patchtst/dataset/ETTh1_blind_test.csv`.
- Change the physical split policy.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Change the task. `pred_len` must stay `96`.

**The goal is simple: get the lowest val_mse.** Since the time budget is fixed, you do not need to worry about training time — it is always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Blind test rule**: `blind_test_mse` is recorded for transparency, but it is **not** the branch-advance metric. The branch advances only on `val_mse`. Do not optimize directly against the blind test file.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_mse gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that is a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_mse improvement that adds 20 lines of hacky code is probably not worth it. A 0.001 val_mse improvement from deleting code is definitely worth keeping. An improvement of ~0 but much simpler code is also worth keeping.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_mse:          0.405305
blind_test_mse:   0.431455
training_seconds: 300.0
total_seconds:    303.3
num_steps:        1753
num_epochs:       36
stop_reason:      time_budget
blind_test_file:  /abs/path/patchtst/dataset/ETTh1_blind_test.csv
summary_path:     /abs/path/runs/patchtst/.../run_summary.txt
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metrics from the log file with:

```bash
grep "^val_mse:\|^blind_test_mse:\|^training_seconds:\|^summary_path:" run.log
```

The same summary is also written to `run_summary.txt` inside the run result directory, so you can inspect it even if you did not redirect stdout.

The most important checks are:

- `val_mse`
- `blind_test_mse`
- `blind_test_file` must end with `ETTh1_blind_test.csv`

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```tsv
commit	val_mse	blind_test_mse	status	description
```

1. git commit hash (short, 7 chars)
2. `val_mse` achieved (e.g. `0.405305`) — use `NA` for crashes
3. `blind_test_mse` achieved (e.g. `0.431455`) — use `NA` for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```tsv
commit	val_mse	blind_test_mse	status	description
83ca89d	0.405305	0.431455	keep	blind-test baseline after physical split
abc1234	0.398210	0.440991	keep	reduce dropout and widen feed-forward
def5678	0.412003	0.428100	discard	increase patch stride to 16
987zyxw	NA	NA	crash	double width caused OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar27` or `autoresearch/mar27-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit you are on.
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use `tee` or let output flood your context)
5. Read out the results: `grep "^val_mse:\|^blind_test_mse:\|^summary_path:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 80 run.log` to read the Python stack trace and attempt a fix. If you cannot get things to work after more than a few attempts, give up on that idea.
7. Record the results in the tsv
8. If `val_mse` improved (lower), you "advance" the branch, keeping the git commit
9. If `val_mse` is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they do not, discard. And you are advancing the branch so that you can iterate. If you feel like you are getting stuck in some way, you can rewind but you should do this very very sparingly (if ever).

**Important**: `blind_test_mse` is not the keep/discard metric. It is only logged. This is deliberate so the blind test remains more trustworthy.

**Timeout**: Each experiment should take about 5 minutes total (+ a bit for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it is something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log `crash` as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from the computer and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes about 5 minutes then you can run around 12 per hour, for a total of around 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept.
