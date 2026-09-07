---
layout: post
project: true
title: Don't let an LLM explain your search results
subtitle: Building search that shows its real reasons
tags: [rag, llm, python]
comments: true
share-description: "People don't trust a search result they can't explain. I built the explanation from the ranking signals the search already had, not from an LLM — because an LLM writes a believable reason, not the real one."
---

<!--
  DRAFT 4 of 4. SANITISED: no employer, no product names, no ticket keys, no
  ticket text.
-->

A search system has a trust problem that has nothing to do with ranking quality.

Someone searches for a problem they are stuck on. The top result has a different
title, about a different component, and does not obviously match anything they typed.
It might be a great match. The vector side may have seen that it is the same story
underneath. But the person reading it cannot tell. So they decide the search is broken
and stop using it.

Ranking well is not enough. The system has to say *why*.

## The obvious approach, and why I said no

The obvious move in 2026 is to send the query and the result to an LLM and ask it to
explain the match. I already had a quantised model loaded. It would have taken an
afternoon.

I did not do it, for two reasons.

The first is speed. This runs on one 4 GB GPU. An LLM call in the search path adds
seconds to every query. Search has to feel instant or people stop using it.

The second reason is the one that decided it. **An LLM asked to explain a match will
always write a convincing explanation, whether or not it is the real one.** Give it a
query and a document and it will find some connection between them. That is what it is
good at. But the question I need answered is not "how are these two texts related". It
is "why did *this ranking function* put this document first". Those are different
questions. Only one of them is true.

An explanation that sounds right but describes reasoning the system never did is worse
than no explanation. It teaches people to trust the system for the wrong reasons. And
it fails exactly when they need it most — on the surprising results, which are the ones
they came to check.

## Explaining from signals you already have

By the time the results are ready, the search has already produced everything an
honest explanation needs:

- which fields BM25 matched, and where
- which query keywords overlap the document's keywords
- whether the document was found by BM25, by the vector legs, or by both
- how much each leg added to the final score

So the explainer reads those signals and writes a sentence. No model call. No extra
latency. And nothing it says can be false, because every claim comes from a number the
ranker actually computed.

One detail mattered more than I expected: stopwords. In an issue tracker, words like
*error*, *issue*, *fail*, *problem* and *ticket* show up in almost every document.
"Matched on: error, issue" is true and useless. Worse than useless — it makes the
system look naive. So those words are filtered out of the explanation, even though
they still count toward the score. What is left is the vocabulary that actually
separates this ticket from the other 100,000.

## Making the score mean something

The same trust problem shows up in the percentage next to each result.

The obvious way is to show the cosine similarity. It is a number between 0 and 1.
Multiply by 100 and call it "% match". But e5 embeddings put almost all real cosine
values in a narrow band, roughly 0.7 to 1.0. So every result shows up between 87% and
100%, a couple of points apart. The number tells you nothing.

It can also disagree with the ranking. The order comes from the merged RRF score, not
from cosine. So the top result can show a lower percentage than the one below it.
Users see that and think the search is broken. They are right to.

The fix was to base the percentage on the score that actually sets the order. The top
result is 100%, and everything else is shown relative to it. Now the number always
agrees with the ranking. And documents found only by BM25, which have no vector score
at all, no longer show "0% match" while sitting near the top.

## Weighting the legs by what they mean

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

These weights come from what the fields *are*, not from tuning. Someone searching is
describing a problem they have not solved. So their text looks a lot like a `problem`
section and only a little like a `solution`. Equal weights would let a ticket rank
high just because its fix uses the same words as your symptom. That is exactly the
confusing result that started all of this.

All five rankings go out in one `_msearch`. Running them one after another would cost
a round trip per field.

## The principle

It all comes back to one idea: **the explanation has to come from the mechanism, not
from a model asked to imagine one.**

The percentage comes from the score that sets the order. The match reasons come from
the ranking signals. The weights come from what the fields mean. None of it is
generated, so none of it can drift away from what the system actually did.

That is the only kind of "explainable" that survives a user checking a surprising
result.

<!--
  TODO before publishing:
    - [ ] read aloud
    - [ ] confirm the weights against vector_index.py weights()
    - [ ] link posts 1-3
    - [ ] move to _posts/YYYY-MM-DD-explainable-retrieval.md
-->
