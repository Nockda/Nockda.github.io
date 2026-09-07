---
layout: post
project: true
title: Don't let an LLM explain your search results
subtitle: Designing retrieval that can show its own reasons
tags: [rag, llm, python]
comments: true
share-description: "Users don't trust a search result they can't explain. I built the explanation from the ranking signals the search already produced, not from an LLM — because an LLM writes a plausible reason, not the real one."
---

<!--
  DRAFT 4 of 4. SANITISED: no employer, no product names, no ticket keys, no
  ticket text.
-->

A retrieval system has a trust problem that has nothing to do with relevance.

Someone searches for a problem they are stuck on. The top result is a ticket with a
completely different title, about a different component, that does not obviously
relate to anything they typed. It might be an excellent match — the vector side saw
that it is the same underlying story. But the person looking at it has no way to know
that, so they conclude the search is broken and stop using it.

Ranking correctly is not enough. The system has to be able to say *why*.

## The obvious approach, and why I rejected it

The obvious move in 2026 is to hand the query and the result to an LLM and ask it to
explain the match. I already had a quantised model loaded. It would have taken an
afternoon.

I did not do it, for two reasons.

The first is cost. This runs on a single 4 GB GPU. Putting an LLM call in the search
path adds seconds to every query, and search has to feel instant or people stop using
it.

The second reason is the one that actually settled it. **An LLM asked to explain a
match will always produce a convincing explanation, whether or not it is the real
one.** Give it a query and a document and it will find a thematic connection between
them, because that is what it is good at. But the thing I need to explain is not
"how are these two texts related" — it is "why did *this ranking function* put this
document first". Those are different questions, and only one of them is true.

An explanation that sounds right but describes reasoning the system never did is
worse than no explanation. It teaches people to trust the system for the wrong
reasons, and it fails exactly when they need it most — on the surprising results,
which are the ones they came to check.

## Explaining from the signals you already have

By the time results are ready, the search has produced everything an honest
explanation needs:

- which fields the BM25 highlighter matched, and where
- which query keywords overlap the document's keywords
- whether this document was found by BM25, by the vector legs, or by both
- each leg's contribution to the fused score

So the explainer reads those signals and turns them into a sentence. No model call,
no latency, and nothing it says can be untrue — every claim traces to a number the
ranker actually computed.

One detail that mattered more than expected: stopwords. In an issue tracker, words
like *error*, *issue*, *fail*, *problem* and *ticket* appear in nearly every
document. "Matched on: error, issue" is technically accurate and completely useless —
worse than useless, because it makes the system look naive. Those terms are filtered
out of the explanation even though they contribute to the score, so what surfaces is
the vocabulary that actually distinguishes this ticket from the other 100,000.

## Making the score mean something

The same trust problem shows up in the percentage next to each result.

The intuitive implementation is to show the cosine similarity. It is a number between
0 and 1, multiply by 100, call it "% match". Except e5-family embeddings put nearly
all real cosine values in a narrow band around 0.7 to 1.0, so every result renders as
somewhere between 87% and 100%, separated by a couple of points. The number
discriminates nothing.

Worse, it can contradict the ranking. Ordering comes from the fused RRF score, not
from cosine, so a document ranked first can display a lower percentage than the one
below it. Users read that as a broken system, and they are not wrong to.

The fix was to normalise against the ranking that actually decides the order: the top
result's fused score is 100%, and everything else is shown relative to it. The number
is now guaranteed to agree with the ordering, and documents found only by BM25 — which
have no vector score at all — no longer display as "0% match" while sitting near the
top.

## Weighting the legs by what they mean

The fusion combines five rankings, and the weights are deliberately asymmetric:

| Ranking | Weight | Reasoning |
| :--- | ---: | :--- |
| `problem` vector | 1.5 | The query is always an unsolved problem. Same kind of text. Strongest signal. |
| `bm25` | 1.0 | Exact terms — error strings, component names, version numbers |
| `root_cause` vector | 1.0 | Fires when the underlying cause is shared |
| `solution` vector | 0.7 | A solution restates its problem, so it matches, but weakly |
| `legacy` vector | 0.5 | Fallback for documents not yet migrated to the axes |

These come from what the fields *are*, not from a sweep. Someone searching is
describing a problem they have not solved, so their text resembles a `problem`
section closely and a `solution` section only incidentally. Weighting them equally
would let a ticket rank highly because its fix happens to share vocabulary with your
symptom — which is precisely the confusing result that started this whole line of
work.

All five rankings go out in a single `_msearch`. Running them sequentially would have
cost a round trip per axis, and the axes are the reason the design works at all.

## The principle

Every part of this comes back to the same idea: **the explanation has to come from
the mechanism, not from a model asked to imagine one.**

The percentage is derived from the score that does the ordering. The match reasons
come from the ranking signals. The weights come from what the fields mean. None of it
is generated, so none of it can drift away from what the system actually did — and
that is the only version of "explainable" that survives contact with a user checking
a surprising result.

<!--
  TODO before publishing:
    - [ ] read aloud
    - [ ] confirm the weight table against vector_index.py weights()
    - [ ] link posts 1-3
    - [ ] move to _posts/YYYY-MM-DD-explainable-retrieval.md
-->
