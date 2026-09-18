# Security

## Reporting something

Use GitHub's private vulnerability reporting: the **Security** tab of this
repository, then **Report a vulnerability**. It stays private until there is a
fix.

Please do not open a public issue for anything exploitable.

Expect an acknowledgement within a week. This is a spare-time project, so a fix
may take longer than that; you will be told either way, and credited in the
changelog unless you would rather not be.

## Supported versions

The most recent release. There are no backports.

## What this tool touches

Worth knowing, because most of the risk here is not a code vulnerability.

**Your writing stays local.** `dragon build` reads your corpus and writes
`data/*.jsonl` inside the project. Nothing is uploaded unless you run
`dragon publish`. `data/`, `adapters/`, `fused/` and `config.yaml` are all in
`.gitignore`; leave them there.

**The endpoint has no authentication.** `dragon serve` binds to `127.0.0.1` on
purpose. Anything reachable on your network can use a model bound to
`0.0.0.0`, so do not move it there and assume nobody will find it.

**Hugging Face tokens** are read from `~/.cache/huggingface/token`, where
`huggingface_hub` puts them. This project neither stores nor transmits them
anywhere else. Use a read token where a read token will do, and revoke what you
are not using.

**Publishing is the irreversible step.** `dragon publish` creates private
repositories, and `--public` requires a typed confirmation. A model that has
been downloaded cannot be recalled by deleting the repository.

**The model card is generated** from your `config.yaml` and the adapter's
config. It includes your system prompt, because a user of the model needs it.
If your prompt contains something you would not publish, change the prompt.

**Base model weights come from the Hugging Face Hub** and are loaded by
`mlx-lm`, which this project calls as a subprocess or a library. Model files
are executable content in the sense that matters: pull them from accounts you
have reason to trust.

## Out of scope

Misuse of a correctly functioning tool is not a vulnerability, but it is still
worth reading [docs/responsible-use.md](docs/responsible-use.md). Impersonation
is the risk this project is built around, and the defaults are set accordingly.
