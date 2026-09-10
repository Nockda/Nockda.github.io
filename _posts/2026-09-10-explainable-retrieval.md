---
layout: post
project: true
title: Don't let an LLM explain your search results
subtitle: Why the explanation has to come from the ranking, not from a model
tags: [rag, llm, python]
comments: true
thumbnail-img: /assets/img/explainable/reason.jpg
share-img: /assets/img/explainable/reason.jpg
share-description: "A model asked to explain a search result will always write a convincing reason, whether or not it is the real one. I built the explanation out of the ranking signals instead. Here is what that took."
---

## The situation

Someone searches my issue tracker for a bug they are stuck on. The top result has a
different title, about a different component, and does not obviously match anything
they typed.

It might be an excellent match. The vector side may have seen that it is the same
story underneath, described in different words. That is exactly what vector search is
for.

But the person reading it cannot tell. All they see is a result that looks wrong
sitting in first place. So they decide the search is broken and go back to asking a
colleague.

This is a trust problem, and it is separate from ranking quality. I could improve the
ranking forever and not fix it. A result nobody believes is worth about as much as a
result that never came back.

## The task

The system had to say *why* each result ranked where it did. Three constraints on
that:

1. **It has to be fast.** Search is used mid-debugging. If it stops feeling instant,
   people go back to asking a colleague and the whole thing is dead.
2. **It has to be true.** Not plausible. True. The explanation is being used to decide
   whether to trust a surprising result, which is the one case where a wrong
   explanation does real damage.
3. **The number next to it has to agree with it.** Showing "94% match" on a result
   ranked below one showing 91% undoes any amount of good explanation.

## The action

### Why I did not ask the model

The obvious move in 2026 is to hand the query and the document to an LLM and ask why
they match. I already had a quantised model loaded. It would have taken an afternoon.

The speed constraint rules it out on its own. An LLM call in the search path adds
seconds to every query, and the GPU is shared with other services.

But the second reason is the one that actually decided it. **A model asked to explain
a match will always write a convincing explanation, whether or not it is the real
one.** Give it a query and a document and it will find a connection between them.
That is what it is good at.

The question I need answered is not "how are these two texts related". It is "why did
*this ranking function* put this document first". Those are different questions, and a
model only ever answers the first one.

An explanation that sounds right but describes reasoning the system never did is
worse than no explanation at all. It teaches people to trust the system for reasons
that are not the reasons. And it fails hardest on surprising results, which are the
only ones anybody checks.

### Building the explanation out of the ranking

By the time the results are ready, the search has already produced everything an
honest explanation needs:

- which fields BM25 matched, and where
- which query keywords overlap the document's keywords
- whether the document was found by BM25, by the vector legs, or by both
- how much each leg contributed to the final score

So the explainer reads those signals and writes a sentence from them. No model call.
No extra latency. And nothing it says can be false, because every claim traces back to
a number the ranker actually computed.

One detail mattered more than I expected: stopwords. In an issue tracker, words like
*error*, *issue*, *fail*, *problem* and *ticket* appear in almost every document.
"Matched on: error, issue" is completely true and completely useless. Worse than
useless, because it makes the system look naive at the exact moment it is trying to
earn trust. So those words are filtered out of the explanation, even though they still
count toward the score. What is left is the vocabulary that actually separates this
ticket from the other 100,000.

### Making the number agree with the order

The obvious way to show a match percentage is to take the cosine similarity, multiply
by 100, and print it.

That fails twice. First, e5 embeddings put almost all real cosine values in a narrow
band, roughly 0.7 to 1.0. Every result comes out between 87% and 100%, a couple of
points apart. The number cannot discriminate, so it tells the reader nothing.

Second, and worse, it can disagree with the ranking. The order comes from the merged
RRF score, not from cosine. So the top result can display a lower percentage than the
one beneath it. A user who sees that concludes the search is broken, and they are
right to.

So the percentage is now derived from the score that actually sets the order. The top
result is 100%, everything else is relative to it. The number agrees with the ranking
by construction. And a document found only by BM25, which has no vector score at all,
no longer shows "0% match" while sitting near the top.

### Weighting the legs by what they mean

The merge combines five rankings, and the weights are deliberately uneven:

<svg viewBox="0 0 720 300" role="img" width="100%"
     aria-label="Five rankings feed one weighted reciprocal rank fusion: problem vector at weight 1.5, bm25 at 1.0, root_cause vector at 1.0, solution vector at 0.7, legacy vector at 0.5. All five are issued in a single msearch round trip and produce one ranked list."
     style="max-width:720px;margin:2rem auto;display:block;font-family:'Open Sans',Helvetica,Arial,sans-serif">

  <text x="152" y="26" text-anchor="end" font-size="11.5" letter-spacing="1.2" fill="var(--ink-soft)">RANKING</text>
  <text x="164" y="26" font-size="11.5" letter-spacing="1.2" fill="var(--ink-soft)">WEIGHT</text>
  <text x="360" y="26" font-size="11.5" letter-spacing="1.2" fill="var(--ink-soft)">WHY</text>

  <text x="152" y="56" text-anchor="end" font-size="12" font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink)" font-weight="700">problem vector</text>
  <rect x="164" y="45" width="150.0" height="14" rx="2" fill="var(--accent)" opacity="0.95"/>
  <text x="322.0" y="56" font-size="11.5" font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink-soft)">1.5</text>
  <text x="360" y="56" font-size="11.5" fill="var(--ink-soft)">same kind of text as the query</text>
  <path d="M336 52 L500 52" stroke="none"/>
  <text x="152" y="88" text-anchor="end" font-size="12" font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink-soft)" font-weight="400">bm25</text>
  <rect x="164" y="77" width="100.0" height="14" rx="2" fill="var(--accent)" opacity="0.45"/>
  <text x="272.0" y="88" font-size="11.5" font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink-soft)">1.0</text>
  <text x="360" y="88" font-size="11.5" fill="var(--ink-soft)">exact terms, error strings</text>
  <path d="M336 84 L500 84" stroke="none"/>
  <text x="152" y="120" text-anchor="end" font-size="12" font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink-soft)" font-weight="400">root_cause vec</text>
  <rect x="164" y="109" width="100.0" height="14" rx="2" fill="var(--accent)" opacity="0.45"/>
  <text x="272.0" y="120" font-size="11.5" font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink-soft)">1.0</text>
  <text x="360" y="120" font-size="11.5" fill="var(--ink-soft)">shared underlying cause</text>
  <path d="M336 116 L500 116" stroke="none"/>
  <text x="152" y="152" text-anchor="end" font-size="12" font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink-soft)" font-weight="400">solution vector</text>
  <rect x="164" y="141" width="70.0" height="14" rx="2" fill="var(--accent)" opacity="0.45"/>
  <text x="242.0" y="152" font-size="11.5" font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink-soft)">0.7</text>
  <text x="360" y="152" font-size="11.5" fill="var(--ink-soft)">restates its problem, weakly</text>
  <path d="M336 148 L500 148" stroke="none"/>
  <text x="152" y="184" text-anchor="end" font-size="12" font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink-soft)" font-weight="400">legacy vector</text>
  <rect x="164" y="173" width="50.0" height="14" rx="2" fill="var(--accent)" opacity="0.45"/>
  <text x="222.0" y="184" font-size="11.5" font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink-soft)">0.5</text>
  <text x="360" y="184" font-size="11.5" fill="var(--ink-soft)">not yet migrated</text>
  <path d="M336 180 L500 180" stroke="none"/>

  <!-- bracket collecting the five legs -->
  <path d="M596 45 h12 v144 h-12" fill="none" stroke="var(--border)" stroke-width="1.5"/>
  <text x="614" y="112" font-size="11.5" fill="var(--ink-soft)">one</text>
  <text x="614" y="128" font-size="11.5" font-family="ui-monospace,Menlo,monospace" fill="var(--ink)">_msearch</text>

  <!-- down into fusion -->
  <path d="M602 189 L602 214 Q602 224 592 224 L436 224" fill="none" stroke="var(--ink-soft)" stroke-width="1.5"/>
  <path d="M430 224 l8 -5 v10 z" fill="var(--ink-soft)"/>

  <rect x="248" y="204" width="182" height="40" rx="6" fill="var(--surface)" stroke="var(--border)" stroke-width="1.5"/>
  <text x="339" y="222" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--ink)">weighted RRF</text>
  <text x="339" y="237" text-anchor="middle" font-size="11" font-family="ui-monospace,Menlo,monospace" fill="var(--ink-soft)">rank constant 60</text>

  <path d="M339 244 L339 262" fill="none" stroke="var(--ink-soft)" stroke-width="1.5"/>
  <path d="M339 268 l-5 -9 h10 z" fill="var(--ink-soft)"/>

  <rect x="219" y="274" width="240" height="9" rx="2" fill="var(--accent)"/>
  <rect x="219" y="286" width="196" height="9" rx="2" fill="var(--accent)" opacity="0.7"/>
  <text x="474" y="288" font-size="12" font-weight="700" fill="var(--ink)">one ranked list</text>
</svg>

These come from what the fields *are*, not from a sweep. Someone searching is
describing a problem they have not solved. Their text looks a lot like a `problem`
section and only a little like a `solution`. Equal weights would let a ticket rank
high just because its fix uses the same words as your symptom. That is the confusing
result this whole thing exists to prevent.

All five go out in one `_msearch`. Running them one after another would cost a round
trip per field.

## The result

Before this, I measured how often the displayed percentage disagreed with the order it
was printed next to. **Nine of thirteen searches, 69%.** Within the top five, the
median gap between two results was 2.2 points, so the number was not discriminating
anything either.

Both of those are now structurally impossible rather than fixed. The first result is
100% because the scale is defined that way, and every other result is a fraction of
the score that ordered it. There is no arrangement of inputs that puts them out of
step.

The explanations cost no measurable latency, because they are string formatting over
numbers the search already had.

And the part I did not anticipate: making the weights explicit made them arguable. The
`problem` leg is at 1.5 because I can say out loud why it deserves to be, and someone
can tell me I am wrong. That is a better position than having the behaviour fall out of an
implementation detail nobody ever looked at.

## What I would take from this

If you are adding explanations to a retrieval system, the useful question is not
"which model should write them". It is **"what does my ranker already know that it is
throwing away".**

Almost always the answer is: most of it. The per-field matches, the per-leg
contributions, which retriever found the document. All of that exists for a few
milliseconds inside the search and then gets discarded so a single sorted list can be
returned.

An explanation built out of those numbers cannot drift from what the system did. An
explanation generated after the fact always can, and you will not be able to tell the
difference by reading it. That is the whole problem.
