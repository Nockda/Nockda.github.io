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
