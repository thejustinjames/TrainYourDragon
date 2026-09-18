# Voice and knowledge

The single most useful thing to understand before you start: fine-tuning on
your own writing teaches the model how you write. It does not teach it what you
know. Those are different problems, and only one of them is solved here.

## What actually happens in a run

A LoRA fine-tune adjusts a small number of weights — a rank-8 adapter on the
attention and feed-forward projections of the top layers is around eleven
million parameters against a base model's seven billion, about 0.15% of it. It
is a nudge, not a rebuild.

What a nudge that size can encode is style. Sentence length. Where you put a
full stop instead of a comma. Whether you open with the claim or circle to it.
Whether you use bullets. Which adjectives you never use. The shape of a
paragraph, and the shape of an argument across paragraphs. All of that is
consistent across everything you have written, so it is the pattern the model
finds first and holds most strongly.

What a nudge that size cannot encode is a body of facts. Facts are sparse: the
name of your product appears in a hundred places, but what it *does* appears
properly in perhaps two, and the model has no reason to privilege those two
over everything it already believed about that word. The base model's prior
wins.

## The Silo problem

The first version of this tooling made a training pair out of every section of
every page, framed like this:

```
Product: Silo — a governed content store
Category: Tools · Status: Shipping
Section: Who it is for

Write this section of the product page.
```

The brief always handed over the name *and* the tagline *and* the description.
So the model never once had to recall what Silo was from the name alone. It
only ever had to continue in the right voice from a brief that already
contained the answer.

Asked afterwards, through the endpoint, "tell me about Silo", it produced a
confident, well-cadenced paragraph describing a content management system. Not
the product. The base model's prior meaning of the word "silo" had never been
challenged, because the training never challenged it.

This is why the builder generates **recall pairs**: a dozen phrasings per page,
asked by name and nothing else, answered from what the corpus actually says
that thing is.

```
What is Silo?
Who is Silo for?
Why did you build Silo?
Give me the one-line pitch for Silo.
```

It helps. It is not a fix. Recall pairs teach the model that a name has a
meaning; they do not give it a reliable memory, and the more you add the more
you are training a lookup table with a bad index, at the cost of the voice you
actually wanted.

## Where the line falls

After a normal run — a few hundred thousand words, a couple of passes — expect
roughly this:

| | Learnt |
|---|---|
| Register, cadence, paragraph shape | Yes, strongly, and quickly |
| Vocabulary and the words you avoid | Yes |
| The structure of your arguments | Partly: the shape, not the substance |
| Your habitual moves and asides | Yes, sometimes too well |
| What your products are | Weakly, and confidently wrong when weak |
| Dates, numbers, names, quotations | No. Treat every one as invented |
| Judgement about what is worth saying | No |

The characteristic failure is worth recognising, because it is not obvious at a
glance. Given notes with five real points, the fine-tune writes five good
paragraphs and then a sixth that has your rhythm, your sentence length, your
habit of the two-beat close — and says nothing at all. It learned that a
section of yours usually runs six paragraphs. It has nothing to put in the
sixth. So it pads, in your voice, which is much harder to spot than padding in
someone else's.

## What to do about the knowledge half

Don't solve it with fine-tuning. The tools that work:

**Put the facts in the brief.** The model is very good at writing well from
material it is handed. `dragon write --notes` exists for exactly this: your
notes carry the substance, the adapter carries the voice.

**Retrieval, if you need answers rather than drafts.** Search your own corpus,
put the passages in the prompt, and let the model write from them. That is a
different tool from this one and it should be: it can cite, and this cannot.

**Recall pairs, in moderation.** Turn them on, cap them with a number rather
than `true` if your corpus is small, and treat them as a way of stopping the
model being confidently wrong about names — not as a memory.

**Read everything before it leaves the building.** A draft in your own voice
is unusually easy to approve without reading. That is the one real risk of a
tool like this, and it is a discipline problem, not a technical one.

## The useful way to think about it

The model is a very good impression of your writing with nothing behind it. All
the judgement, all the facts, and everything worth saying still has to come
from you. What you get back is the hour spent turning six bullet points into
four paragraphs that sound right.

That is a real hour, most days. It is not the hour anyone imagined an AI would
save, and it is worth being clear-eyed that it is the one on offer.
