# Responsible use

Short, because the problem is simple and the answer is mostly common sense.

## What this tool makes

A model that writes in a specific, identifiable person's voice, well enough
that people who know that person's writing will not always tell. That is the
stated goal. It is also the entire risk, and there is no version of the tool
where it is one without being the other.

## Train on your own writing

The clean case is your own published work, trained by you, kept by you. Nothing
below complicates it.

Training on someone else's writing needs their agreement, in advance,
specifically for this. "It was on the public internet" is not agreement, and in
most places it is not a defence either — copyright and personality rights apply
whatever the robots.txt said. A corpus of a colleague's essays, an author you
admire, or a client's blog is not yours to train on because you could read it.

For a house voice drawn from several people's work, ask each of them, tell them
what it will be used for, and tell them where it will live.

## Keep it private unless you have a reason not to

The defaults here are private repositories and a typed confirmation before
anything goes public, because publishing is the step that cannot be undone. A
downloaded model stays downloaded. Deleting the repository afterwards achieves
nothing.

If you do publish, publish the adapter rather than the fused model: it is
useless without the base weights, which is a small but real barrier. Consider
gating it so requests are approved by hand.

## Say that it is a machine draft

If something the model wrote goes out under your name, you read it, you
corrected it and you are answerable for it — which is the arrangement that
makes it fine. That is a different thing from passing off unread output as
considered work, and the difference is not visible to the reader, so the
obligation sits with you.

Where you are subject to disclosure rules — a publication's policy, a client
contract, a regulator, a journal — a model trained on your own voice is not a
loophole in them. It is exactly the thing they are about. The fact that the
output sounds like you is not a defence; it is the mechanism.

## What it will get wrong

It invents facts fluently and in your register, which makes them harder to
catch than ordinary machine slop. Names, numbers, dates, quotations and
anything about your own products are decoration until you have checked them.
See [voice and knowledge](voice-and-knowledge.md).

The practical failure mode is not the model lying. It is you approving a draft
without reading it because it sounds right.

## Do not use it to impersonate

Explicitly: do not use this to write as someone else in a way that could be
taken for them. Not as a joke, not as a test, not as a demo. That includes
correspondence, social posts, reviews, and anything that could reach a person
who would act on it believing it came from the named author.

If you are building something where a voice model might be mistaken for a
person, put the disclosure where the reader is, not in a footer.

## Things worth doing

- Keep the training data on your machine. This repository's `.gitignore` is set
  up for that; do not undo it.
- Hold pieces back from training, so you can measure rather than guess.
- Record which corpus a published adapter was trained on. The model card
  records the configuration, not the consent.
- Revoke Hugging Face tokens you are not using.
- If you ever train on someone else's writing with their agreement, write down
  what they agreed to, and honour a request to delete it.
