# Contributing

Issues and pull requests are welcome. It is a small project with a narrow
purpose, so the most useful thing you can do before writing code is open an
issue and say what you are trying to do.

## Getting set up

```bash
git clone https://github.com/thejustinjames/TrainYourDragon.git
cd TrainYourDragon
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'          # add ,mlx on Apple silicon
pytest
ruff check src tests
ruff format --check src tests
```

The tests do not need mlx-lm, a GPU or a corpus. They run against the tiny
example in `examples/`, which is why that exists. If your change cannot be
tested without training something, it probably needs restructuring so that the
part worth testing does not.

## What tends to get merged

- A new source kind, if it covers a shape of writing the existing five do not.
- Support for a base model family whose chat template or stop tokens differ.
- Better handling of real-world markup in the cleaner, with a test showing the
  case that was breaking.
- Documentation that corrects something wrong, or covers something a first-time
  user got stuck on.
- Bug fixes, with a test that fails before and passes after.

## What tends not to

- Anything that makes it easier to train on writing that is not yours, or to
  publish without noticing. The defaults here are deliberate; see
  [docs/responsible-use.md](docs/responsible-use.md).
- A training framework other than mlx-lm. This project is deliberately one
  thing: a LoRA on Apple silicon. If you want the same idea on CUDA, fork it.
- A web interface.
- Dependencies. The core needs PyYAML and nothing else, and that is worth
  keeping.

## House style

**Python.** Ruff for both linting and formatting, at 100 columns, configured in
`pyproject.toml`. Type hints on anything public. No comment that restates the
line below it; comments are for why, not what.

**Prose.** British English. Plain and short. Say the thing and stop. The
documentation is written to be read by someone who is about to spend six hours
training a model and would like to not waste them, so it is direct about what
does not work.

**Commits.** A sentence in the imperative, describing the change. No prefixes,
no emoji.

## Pull requests

Keep them to one thing. Say what problem it solves and how you know it works.
If it changes behaviour, update the documentation in the same pull request —
`docs/` is part of the deliverable here, not an afterthought.

CI runs the tests and the linter on Linux against Python 3.10 through 3.13.
Anything that needs Apple silicon has to be verified by hand; say in the pull
request that you did.

## Releases

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com), and
versions follow [semantic versioning](https://semver.org). Add your change
under **Unreleased**.
