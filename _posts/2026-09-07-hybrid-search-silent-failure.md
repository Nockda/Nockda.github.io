---
layout: post
project: true
title: The silent failure mode in hybrid search
subtitle: When one leg returns nothing, the results still look fine
tags: [rag, llm, python]
comments: true
share-img: /assets/img/hybrid-search/share.png
share-description: "Hybrid search merges BM25 and vector results. If one side returns nothing, the output still looks fine. How I found this in a 100,000-ticket search system and fixed it."
---

Hybrid search is the standard way to do retrieval now. BM25 finds exact words. Vector
KNN finds similar meaning. You run both, then merge the two rankings with reciprocal
rank fusion (RRF). It is in every tutorial, and it works.

But it can break in a way nothing warns you about. **One of the two legs can return
nothing, and the merged result still looks fine.**

I hit this in a search system I built over an internal issue tracker. About 100,000
tickets. Engineers use it to find the ticket where someone already fixed the same
problem. Here is what went wrong, and how I fixed it.

## The setup

In production this runs on an **A100 with 80 GB of VRAM**. But I develop it on a
laptop with an **RTX A1000 and 4 GB**, and I wanted the whole system to run there too.
The tighter machine is what shaped the model choices:

- **Gemma-4 E2B-it**, 4-bit nf4 quantised with bitsandbytes. It structures tickets and
  reads image attachments. Quantising it keeps it inside 4 GB, and on the A100 it
  leaves room for other services on the same card.
- **multilingual-e5-large** for embeddings, 1024 dimensions
- **Elasticsearch 9.x** for both the keyword index and the vectors

When a ticket is added, the model splits it into fields: problem, root cause,
solution, keywords. When someone searches, BM25 and vector KNN both run, and RRF
merges the two rankings.

## The trap: long queries

One search path does not use a short phrase. You upload a ticket, the model summarises
it, and that whole summary becomes the query. These summaries are long. The median is
**866 tokens**.

Now look at this BM25 setting. It looks completely normal:

```python
"minimum_should_match": "30%"
```

For "login fails after update" it is right. Six words, two must match.

For an 866-token query it matches nothing. `"30%"` of a 300-word query means a
document must share **90 words** before Elasticsearch will even look at it.

<svg viewBox="0 0 720 350" role="img" width="100%"
     aria-label="Two rankings feed a reciprocal rank fusion step. The BM25 ranking is empty while the vector ranking has five results. The fused output still shows five ranked results, so the failure is invisible from the output."
     style="max-width:720px;margin:2rem auto;display:block;font-family:'Open Sans',Helvetica,Arial,sans-serif">

  <!-- BM25 leg: empty -->
  <rect x="30" y="34" width="280" height="132" rx="6"
        fill="none" stroke="var(--border)" stroke-width="1.5"/>
  <text x="46" y="58" font-size="13" font-weight="700" fill="var(--ink)">BM25</text>
  <text x="46" y="76" font-size="11.5" fill="var(--ink-soft)">minimum_should_match: 30%</text>
  <rect x="46" y="90" width="248" height="58" rx="4"
        fill="none" stroke="var(--ink-soft)" stroke-width="1.5" stroke-dasharray="5 4"/>
  <text x="170" y="124" text-anchor="middle" font-size="13" font-weight="700"
        fill="var(--ink-soft)">0 results</text>

  <!-- vector leg: populated -->
  <rect x="410" y="34" width="280" height="132" rx="6"
        fill="none" stroke="var(--border)" stroke-width="1.5"/>
  <text x="426" y="58" font-size="13" font-weight="700" fill="var(--ink)">vector KNN</text>
  <text x="426" y="76" font-size="11.5" fill="var(--ink-soft)">multilingual-e5-large</text>
  <rect x="426" y="90"  width="240" height="10" rx="2" fill="var(--accent)"/>
  <rect x="426" y="104" width="206" height="10" rx="2" fill="var(--accent)" opacity="0.85"/>
  <rect x="426" y="118" width="178" height="10" rx="2" fill="var(--accent)" opacity="0.7"/>
  <rect x="426" y="132" width="150" height="10" rx="2" fill="var(--accent)" opacity="0.55"/>
  <rect x="426" y="146" width="124" height="10" rx="2" fill="var(--accent)" opacity="0.4"/>

  <!-- both legs route down into the fusion box -->
  <path d="M170 172 L170 196 Q170 206 180 206 L360 206 L360 214"
        fill="none" stroke="var(--ink-soft)" stroke-width="1.5" stroke-dasharray="5 4"/>
  <path d="M550 172 L550 196 Q550 206 540 206 L360 206"
        fill="none" stroke="var(--accent)" stroke-width="1.5"/>
  <path d="M360 222 l-5 -9 h10 z" fill="var(--ink-soft)"/>

  <!-- fusion -->
  <rect x="270" y="222" width="180" height="40" rx="6"
        fill="var(--surface)" stroke="var(--border)" stroke-width="1.5"/>
  <text x="360" y="247" text-anchor="middle" font-size="13" font-weight="700"
        fill="var(--ink)">RRF fusion</text>

  <!-- output -->
  <path d="M360 262 L360 286" fill="none" stroke="var(--ink-soft)" stroke-width="1.5"/>
  <path d="M360 292 l-4 -8 h8 z" fill="var(--ink-soft)"/>
  <rect x="240" y="298" width="240" height="10" rx="2" fill="var(--accent)"/>
  <rect x="240" y="312" width="206" height="10" rx="2" fill="var(--accent)" opacity="0.85"/>
  <rect x="240" y="326" width="178" height="10" rx="2" fill="var(--accent)" opacity="0.7"/>
  <text x="500" y="318" font-size="12.5" font-weight="700" fill="var(--ink)">looks completely normal</text>
</svg>

No document shares 90 words. So BM25 returns an empty list. RRF then merges one full
ranking with one empty one. What comes out is plain vector search. It is still called
hybrid.

## Why this is worth knowing

The threshold is not the interesting part. What matters is how this looks from
outside.

The results page is full. The scores look normal. The results are even useful, because
vector search on its own is not bad search. But it stops finding the things BM25 is
there for: an exact error string, a version number, a component name. The embedding
blurs those into similar-looking words.

**A hybrid system with one dead leg looks the same as a healthy one.** No error. No
empty page. No change in speed. All the usual alerts stay green.

If a component returns junk, you notice. If it returns nothing, you do not. Its weight
just moves to whatever else is in the merge.

## The fix

Make the threshold depend on how long the query is:

```python
def _min_should_match(query_text: str) -> str:
    term_count = len(query_text.split())
    if term_count <= 15:
        return "30%"      # short query — the normal default is fine
    if term_count <= 60:
        return "2<-25%"   # 2 words or fewer: match all. Longer: match 25%
    return "4"            # very long — use a fixed count, not a ratio
```

The last line matters most. If you keep using a percentage, longer queries need more
and more matching words. A 2,000-token query at 25% still needs hundreds of them. That
is the same bug again with bigger numbers. At some length you have to stop using a
ratio. Four good word matches in a very long query is already a strong signal.

Elasticsearch supports the `"2<-25%"` form directly. It means: with 2 words or fewer,
match all of them; above that, match 25%. It is in the docs, and almost nobody uses
it.

## How to check your own

Add one line before you merge:

```python
logger.debug("bm25=%d knn=%d", len(bm25_hits), len(knn_hits))
```

Every leg should say how many results it returned. **Zero is the case you care
about**, and it is the one you cannot see in the output.

If you run hybrid search and have never checked these counts on your longest queries,
go and look. It takes a minute.
