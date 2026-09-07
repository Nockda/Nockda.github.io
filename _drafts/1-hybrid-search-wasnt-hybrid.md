---
layout: post
project: true
title: The silent failure mode in hybrid search
subtitle: When one leg returns nothing, the results still look fine
tags: [rag, llm, python]
comments: true
share-description: "Hybrid search fuses BM25 and vector rankings. If one leg quietly returns nothing, the output still looks correct. Here is how I found it in a 100,000-ticket index and fixed it."
---

<!--
  DRAFT 1 of 4. Jekyll ignores _drafts/.
  SANITISED: no employer, no product names, no ticket keys, no ticket text,
  no business metrics. The example query is invented.
-->

Hybrid search is the standard recipe for retrieval now. BM25 for exact terms, vector
KNN for meaning, the two rankings fused with reciprocal rank fusion. It is in every
tutorial and it works.

It also has a failure mode that nothing warns you about: **one of the two legs can
return nothing, and the fused output will still look completely normal.**

I found this in a retrieval system I built over an internal issue tracker — around
100,000 tickets, searched by engineers looking for the ticket where someone already
solved the thing in front of them. Here is how it happens and how I fixed it.

## The setup

Everything runs on a single RTX A1000 with **4 GB of VRAM**, which drove most of the
design:

- **Gemma-4 E2B-it**, 4-bit nf4 quantised through bitsandbytes, for structuring
  tickets and reading image attachments
- **multilingual-e5-large** for embeddings, 1024 dimensions
- **Elasticsearch 9.x** holding both the inverted index and the dense vectors

On ingest, the model turns each raw ticket into structured fields — problem, root
cause, solution, keywords. On search, BM25 and vector KNN both run and get fused.

## The trap: query length

One of the search paths does not take a phrase typed into a box. It takes a ticket
you just uploaded, structures it, and uses **the whole summary** as the query. Median
length: **866 tokens**.

Now look at a completely ordinary BM25 setting:

```python
"minimum_should_match": "30%"
```

For "login fails after update" this is exactly right. Six terms, two must match.

For an 866-token query it excludes the entire corpus. `"30%"` on a 300-term query
means a document has to share **90 terms** with the query before Elasticsearch will
consider it a candidate at all.

Nothing shares 90 terms. The BM25 leg comes back empty, RRF fuses one populated
ranking with one empty one, and the result is pure vector search wearing a hybrid
label.

## Why this is worth knowing about

The interesting part is not the threshold. It is what the failure looks like from
outside.

The results page is full. The scores are reasonable. Relevance is *fine* — vector-only
search is not broken search, it is decent search. It simply stops catching the things
BM25 exists to catch: an exact error string, a version number, a component name that
the embedding smooths into its neighbours.

**A hybrid system running on one leg looks exactly like one running on two.** There is
no exception, no empty state, no latency change. Every signal a service is normally
monitored by stays green.

A component that returns *garbage* announces itself. A component that returns
*nothing* just shifts its weight silently onto whatever else is in the fusion.

## The fix

Scale the threshold to the query instead of hardcoding it:

```python
def _min_should_match(query_text: str) -> str:
    term_count = len(query_text.split())
    if term_count <= 15:
        return "30%"      # short query — the usual default is right
    if term_count <= 60:
        return "2<-25%"   # 2 terms or fewer: all must match. Longer: 25%
    return "4"            # very long — switch from a ratio to a fixed count
```

That last branch is the one people miss. If you keep a percentage all the way up, the
number of required terms grows without bound as the query grows — a 2,000-token query
under a 25% rule still needs hundreds of matches, and you are back to the same bug
with different numbers. Past a certain length you have to stop thinking in ratios.
Four solid term matches out of a very long query is a real signal.

Elasticsearch's `minimum_should_match` supports the `"2<-25%"` form natively: "if
there are 2 or fewer terms require all of them, above that require 25%". It is in the
docs and almost nobody uses it.

## How to check your own

One line, before you fuse:

```python
logger.debug("bm25=%d knn=%d", len(bm25_hits), len(knn_hits))
```

Every leg of a fusion should report how much it contributed, because **zero is the
interesting case** and it is the only one you cannot see from the output. If you are
running hybrid search today and you have never looked at the per-leg counts on your
longest queries, go and look. It takes a minute.

<!--
  TODO before publishing:
    - [ ] read aloud, split anything you stumble on
    - [ ] confirm 866 tokens and the code against the repo
    - [ ] link post 2 once it exists
    - [ ] move to _posts/YYYY-MM-DD-hybrid-search-silent-failure.md
-->
