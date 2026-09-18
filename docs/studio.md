# The observability studio

```bash
bash scripts/studio.sh        # or: dragon studio
```

opens a page at <http://127.0.0.1:8790/> that watches a training run: a
progress bar with a time estimate, the loss curve drawn live, the checkpoints
as they land, and a paragraph in plain words about what the numbers mean right
now. It polls `train.log` every two seconds. It is read-only, so pointing it at
a run that is already going changes nothing about that run.

There is nothing to install and no build step. It is one Python file serving
one HTML file from the standard library. The page carries the Agencie.io Labs
mark and loads its typefaces (Syne, Familjen Grotesk, Space Mono) from Google
Fonts; if that is unreachable it falls back to the system font and nothing else
changes. No training data leaves the machine either way.

## What it shows

**Progress.** Iteration against the run's target, time left from the trainer's
own iterations-per-second, tokens per second, peak memory, and how much of the
model is actually being trained (the adapter's parameters against the base's).

**Loss.** Validation loss at every evaluation, training loss at every report,
a dashed line for every checkpoint saved, and a ring on the lowest validation
point. The first validation figure is enormous — the untouched model asked to
write in a voice it has never seen — so the vertical axis is clamped to keep
the interesting part legible.

**What this means.** Generated from the state, not fixed text: where the low
point is, whether the last point is above it, whether the curve has flattened,
how many passes over the corpus the run has made, and, if the log has gone
quiet, that the Mac is probably dozing and how to stop it. It is meant as much
as a teaching aid as a dashboard: a first-time trainer should be able to read
this panel and know when to stop.

**Corpus.** The makeup of the dataset by kind of pair, from `data/stats.json`.

**Log.** The last dozen lines that say something, with the progress bars
filtered out.

## Previous runs

Every `train*.log` in the project is found and listed as a tab, newest first:
`train.log` is the current run, and anything kept as `train-run1.log`,
`train-before-recall.log` or similar is an earlier one, with
`adapters-run1/` beside it if that was kept too. A finished run reads its
length from its own log, so the tabs work without a config.

**Overlay the other runs** draws every other run's validation curve as a
dashed grey line on the current chart, labelled at its end. That is the
comparison that matters between runs: whether the floor came down, and where
the upturn moved to.

## Watching a run you started by hand

It does not need a `config.yaml`. Any `mlx_lm.lora` log will do:

```bash
dragon studio --log ~/other-project/train.log \
              --adapters ~/other-project/adapters \
              --iters 1600
```

`--iters` gives it the target when there is no config to read one from.
Without it the progress bar has nothing to measure against, and the rest still
works.

## Starting it, and the port

`scripts/studio.sh` is the friendly way in: it finds the virtual environment,
passes anything you give it to `dragon studio`, and starts the server. If port
8790 is already taken, the studio says by what and asks:

```
Port 8790 is already in use by dragon (pid 82513).
That looks like an earlier studio, probably still watching a run.
[e]nd it and use 8790, use [a]nother port (8791), or [q]uit?
```

Type a port number instead and it uses that. When there is no terminal to ask
— a launcher, a script — it takes the next free port and says so. `--if-busy`
sets the behaviour explicitly:

| | |
|---|---|
| `--if-busy ask` | the exchange above (the default at a terminal) |
| `--if-busy next` | quietly move to the next free port (the default otherwise) |
| `--if-busy kill` | end an earlier studio and take the port; anything that is not a studio is left alone and the next port used |
| `--if-busy fail` | stop with an error |

Finding the process needs `lsof`, which macOS and most Linux systems have. Without it the studio can still move to another port, but cannot say what has the current one.

## Options

| | |
|---|---|
| `--port 8790` | another port |
| `--host 127.0.0.1` | leave it. There is no authentication |
| `--no-open` | do not open a browser tab |

## Stages

| Stage | Meaning |
|---|---|
| `waiting` | no `train.log` yet |
| `loading` | the trainer is up but has not reported an iteration |
| `training` | figures arriving |
| `stalled` | the log has not changed for three minutes and the run is not finished |
| `stopped` | an old log that never reached its end: the run was stopped or it crashed |
| `finished` | the final iteration was reached, or the final weights were saved |
