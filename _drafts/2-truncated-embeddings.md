---
layout: post
project: true
title: Measure your token lengths before you tune anything else
subtitle: A 512-token limit was cutting three quarters of my documents
tags: [rag, llm, python]
comments: true
share-description: "Embedding models truncate silently. In a sample of my index, 75.7% of documents ran past the 512-token limit, and the field describing how each problem was fixed was missing from one vector in four."
---

<!--
  DRAFT 2 of 4. SANITISED: no employer, no product names, no ticket keys, no
  ticket text. All numbers measure my own index, not business data.
-->

Every sentence-transformer has a maximum input length. `multilingual-e5-large`, which
I use, stops at **512 tokens**. Everything past that is dropped before the vector is
computed — quietly, with no warning and no error.

That limit is in the model card and everyone reads it. What almost nobody does is
check it against their own corpus.

I did, on a sample of about a thousand tickets out of roughly 100,000:

| | |
| :--- | ---: |
| Documents over 512 tokens | **805 of 1,064 — 75.7%** |
| Longest document | **2,410 tokens** |

Three out of four were being cut. The longest ran to nearly five times the limit.

## The cut is not random

This is what turns a rounding error into a real retrieval problem.

I was building the embedding input by concatenating fields in a fixed order: title,
problem, root cause, solution, keywords, extra notes. Perfectly reasonable. But a
fixed field order plus a fixed cutoff means the truncation always lands in the same
place. It is not spread across the document. It removes **the same fields, from every
document that runs long**.

So I measured which fields actually survived into the vector:

| Field | Present in the vector |
| :--- | ---: |
| `title` | 100% |
| `problem` | 100% |
| `root_cause` | 99.6% |
| **`solution`** | **76.4%** |
| **`problem` / `solution_keywords`** | **47.9%** |
| `additional_info` | 40.2% |

`solution` is the row that matters. In roughly one document in four, **how the
problem was actually fixed did not exist as far as vector search was concerned** —
and that is the field the whole system exists to retrieve. People search because they
want to know how this was solved last time.

The keywords are almost as bad. That is the field the ingest prompt works hardest to
produce, and it was landing in the vector less than half the time.

## Why nothing complained

A truncated document still produces a vector. It is still 1024 dimensions, it still
has a cosine similarity to the query, it still ranks. It is simply a vector for the
first half of the document, and nothing in the pipeline says so.

If the embedder raised on over-length input, this would be a day-one bug. Instead the
library does the reasonable thing and silently trims — which is the right default for
a library and a trap for anyone indexing long documents.

## The fix: stop making one vector per document

The reflex is to chunk by length: 512-token windows, embed each, store them all.

I did something narrower, because these documents already have structure. Rather than
chunking by length, I embed **by field**: one vector for `problem`, one for
`root_cause`, one for `solution`. Each sits comfortably under the limit on its own,
so nothing is cut.

Search then runs a KNN query per axis and fuses those alongside BM25. The query is
embedded once and reused across all three axes, so the extra legs cost nothing at
embedding time.

This turned out better than length-chunking for a reason I did not anticipate.
Because the axes are separate, they can carry **different weights** — and a query is
always someone's unsolved problem, so it should look most like a `problem` field and
least like a `solution`. It also made a "focused" mode possible: search `problem` and
`root_cause` only, so a ticket does not rank highly just because its *solution*
section happens to share vocabulary with your problem.

The trade-off is honest: three vectors per document instead of one, which means a
larger index and more work at ingest. At this corpus size that is a good deal, and
the axis weights would have been impossible with a single blob.

## The takeaway

Before tuning a reranker, before touching chunk sizes, before swapping models: spend
twenty minutes measuring the token-length distribution of your corpus against your
model's limit.

Not the mean. The **percentage over the limit, and the tail.** If a meaningful share
of your documents runs long, you do not have a relevance problem to tune — you have
documents that are partly absent from the index, and no amount of reranking recovers
what was never embedded.

<!--
  TODO before publishing:
    - [ ] read aloud
    - [ ] confirm 805/1,064, 2,410 and the field table against the eval report
    - [ ] link posts 1 and 3
    - [ ] move to _posts/YYYY-MM-DD-measure-token-lengths.md
-->
