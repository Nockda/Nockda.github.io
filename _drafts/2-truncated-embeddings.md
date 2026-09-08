---
layout: post
project: true
title: Measure your token lengths before you tune anything else
subtitle: A 512-token limit was cutting three quarters of my documents
tags: [rag, llm, python]
comments: true
share-description: "Embedding models cut long text without telling you. In my index, 75.7% of documents were too long, and the field saying how each problem was fixed was missing from one vector in four."
---

<!--
  DRAFT 2 of 4. SANITISED: no employer, no product names, no ticket keys, no
  ticket text. All numbers measure my own index, not business data.
-->

Every embedding model has a maximum input length. `multilingual-e5-large`, the one I
use, stops at **512 tokens**. Anything after that is thrown away before the vector is
made. No warning. No error.

That limit is in the model card, and everyone reads it. Almost nobody checks it
against their own documents.

I checked. I measured about a thousand tickets out of roughly 100,000:

| | |
| :--- | ---: |
| Documents over 512 tokens | **805 of 1,064, or 75.7%** |
| Longest document | **2,410 tokens** |

Three out of four were being cut. The longest was almost five times the limit.

## The cut always lands in the same place

This is what turns a small detail into a real problem.

I was building the embedding input by joining fields in a fixed order: title,
problem, root cause, solution, keywords, extra notes. That seemed fine to me. But a fixed order plus a
fixed cutoff means the cut always falls in the same spot. The damage is not spread
around. **The same fields go missing from every long document.**

So I measured which fields actually made it into the vector:

<svg viewBox="0 0 720 268" role="img" width="100%"
     aria-label="Share of each field that survived into the embedding vector: title 100%, problem 100%, root_cause 99.6%, solution 76.4%, keywords 47.9%, additional_info 40.2%. The last three fall well short because the 512-token cut always lands in the same place."
     style="max-width:720px;margin:2rem auto;display:block;font-family:'Open Sans',Helvetica,Arial,sans-serif">
  <text x="238" y="26" text-anchor="end" font-size="11.5" letter-spacing="1.2"
        fill="var(--ink-soft)">FIELD</text>
  <text x="250" y="26" font-size="11.5" letter-spacing="1.2"
        fill="var(--ink-soft)">SHARE THAT REACHED THE VECTOR</text>
  <text x="238" y="60" text-anchor="end" font-size="12.5"
        font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink-soft)"
        font-weight="400">title</text>
  <rect x="250" y="48" width="380" height="16" rx="2"
        fill="var(--surface)" stroke="var(--border)" stroke-width="1"/>
  <rect x="250" y="48" width="380.0" height="16" rx="2"
        fill="var(--accent)" opacity="0.45"/>
  <text x="640" y="60" font-size="12.5"
        font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink-soft)"
        font-weight="400">100.0%</text>
  <text x="238" y="94" text-anchor="end" font-size="12.5"
        font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink-soft)"
        font-weight="400">problem</text>
  <rect x="250" y="82" width="380" height="16" rx="2"
        fill="var(--surface)" stroke="var(--border)" stroke-width="1"/>
  <rect x="250" y="82" width="380.0" height="16" rx="2"
        fill="var(--accent)" opacity="0.45"/>
  <text x="640" y="94" font-size="12.5"
        font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink-soft)"
        font-weight="400">100.0%</text>
  <text x="238" y="128" text-anchor="end" font-size="12.5"
        font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink-soft)"
        font-weight="400">root_cause</text>
  <rect x="250" y="116" width="380" height="16" rx="2"
        fill="var(--surface)" stroke="var(--border)" stroke-width="1"/>
  <rect x="250" y="116" width="378.5" height="16" rx="2"
        fill="var(--accent)" opacity="0.45"/>
  <text x="640" y="128" font-size="12.5"
        font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink-soft)"
        font-weight="400">99.6%</text>
  <text x="238" y="162" text-anchor="end" font-size="12.5"
        font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink)"
        font-weight="700">solution</text>
  <rect x="250" y="150" width="380" height="16" rx="2"
        fill="var(--surface)" stroke="var(--border)" stroke-width="1"/>
  <rect x="250" y="150" width="290.3" height="16" rx="2"
        fill="var(--accent)" opacity="0.95"/>
  <text x="640" y="162" font-size="12.5"
        font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink)"
        font-weight="700">76.4%</text>
  <text x="238" y="196" text-anchor="end" font-size="12.5"
        font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink)"
        font-weight="700">problem/solution_keywords</text>
  <rect x="250" y="184" width="380" height="16" rx="2"
        fill="var(--surface)" stroke="var(--border)" stroke-width="1"/>
  <rect x="250" y="184" width="182.0" height="16" rx="2"
        fill="var(--accent)" opacity="0.95"/>
  <text x="640" y="196" font-size="12.5"
        font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink)"
        font-weight="700">47.9%</text>
  <text x="238" y="230" text-anchor="end" font-size="12.5"
        font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink)"
        font-weight="700">additional_info</text>
  <rect x="250" y="218" width="380" height="16" rx="2"
        fill="var(--surface)" stroke="var(--border)" stroke-width="1"/>
  <rect x="250" y="218" width="152.8" height="16" rx="2"
        fill="var(--accent)" opacity="0.95"/>
  <text x="640" y="230" font-size="12.5"
        font-family="ui-monospace,Menlo,monospace"
        fill="var(--ink)"
        font-weight="700">40.2%</text>
  <text x="250" y="256" font-size="12" fill="var(--ink-soft)">
    Fixed field order + fixed cutoff = the same fields lost from every long document
  </text>
</svg>

Look at `solution`. In about one document in four, **how the problem was fixed was not
in the vector at all**. That is the field the whole system exists to find. People
search because they want to know how someone fixed this last time.

The keywords are almost as bad. That is the field the ingest prompt works hardest on,
and it reached the vector less than half the time.

## Why nothing complained

A cut document still makes a vector. It is still 1024 dimensions. It still has a
similarity score. It still ranks. It is just a vector for the first half of the
document, and nothing says so.

If the embedder threw an error on long input, I would have found this on day one.
Instead the library quietly trims and returns something that looks fine. That is the
right default for a library. It is a trap if your documents are long.

## The fix: stop making one vector per document

The usual answer is to chunk by length. Cut the text into 512-token pieces, embed
each one, store them all. I did something narrower.

My documents already have structure, so instead of chunking by length I embed **by
field**. One vector for `problem`, one for
`root_cause`, one for `solution`. Each one fits under the limit on its own, so nothing
is cut.

Search then runs one KNN query per field and merges those with BM25. The query is
embedded once and reused for all three, so the extra queries cost nothing at embedding
time.

This turned out better than chunking by length, for a reason I did not expect. Because
the fields are separate, they can have **different weights**. A query is always
someone's unsolved problem. So it should look most like a `problem` field, and least
like a `solution`. It also made a "focused" mode possible: search only `problem` and
`root_cause`. That way a ticket does not rank high just because its *solution* happens
to use the same words as your problem.

There is a cost. Three vectors per document instead of one means a bigger index and
more work when adding tickets. At this size that is a good trade. And the field
weights would have been impossible with one big vector.

## The takeaway

Before you tune a reranker, before you change chunk sizes, before you swap models:
spend twenty minutes measuring how long your documents are against your model's limit.

Not the average. The **share that goes over the limit, and how far the longest ones
go**. If a lot of your documents are too long, you do not have a ranking problem to
tune. You have documents that are half missing from the index. No reranker can find
what was never embedded.

<!--
  TODO before publishing:
    - [ ] read aloud
    - [ ] confirm 805/1,064, 2,410 and the chart against the eval report
    - [ ] link posts 1 and 3
    - [ ] move to _posts/YYYY-MM-DD-measure-token-lengths.md
-->
