---
layout: post
project: true
title: The field order in your embedding is a retrieval decision
subtitle: A 512-token limit was cutting three quarters of my documents
tags: [rag, llm, python]
comments: true

share-description: "I joined my fields into one string and embedded it, the way every tutorial does. 75.7% of the documents ran past the model's limit, and the field describing how each problem was fixed fell off the end."
---

This is the second thing I found while going through the search code behind
[my issue tracker]({% post_url 2026-09-07-hybrid-search-silent-failure %}). The first
was a threshold that had quietly switched off half the search. This one is worse,
because it was in the index rather than the query, so no amount of tuning could have
recovered it.

When a ticket comes in, a local model turns it into structured fields: problem, root
cause, solution, keywords. Then I join those into one string and embed it. That line
looks like this, and it is the most ordinary line in the whole system:

```python
embedding_text = " ".join(filter(None, [
    title,
    result.get("problem", ""),
    result.get("root_cause", ""),
    result.get("solution", ""),
    " ".join(result.get("problem_keywords", [])),
    " ".join(result.get("solution_keywords", [])),
    result.get("additional_info", ""),
]))
```

`multilingual-e5-large` stops at **512 tokens**. Anything past that is thrown away
before the vector is made. No warning, no error.

That limit is in the model card and everyone reads it. I had never checked it against
my own documents. So I measured about a thousand tickets out of roughly 100,000:

| | |
| :--- | ---: |
| Documents over 512 tokens | **805 of 1,064, or 75.7%** |
| Longest document | **2,410 tokens** |

Three out of four were being cut. The longest was almost five times the limit.

## The cut always lands in the same place

If the truncation hit a random part of each document, this would be a quality problem
you could live with. It does not. A fixed field order plus a fixed cutoff means the
cut lands in the same place every time. **The same fields go missing from every long
document.**

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

Read that as a list and it looks like damage. Read it against the `join` above and it
is something else: the numbers fall in exactly the order the fields are concatenated.
`title` first and always present. `additional_info` last and mostly gone.

Look at `solution`. In about one document in four, **how the problem was fixed was not
in the vector at all.** That is the field the whole system exists to find. People
search because they want to know how someone fixed this last time.

The keywords are almost as bad. That is the field the ingest prompt works hardest on,
and it reached the vector less than half the time.

## Why the order was what it was

Nobody chose it. That is the part worth sitting with.

The order in the `join` is the order the fields come out of the summariser's output
schema. The model returns problem, then root cause, then solution, then keywords, and
the join walks that dictionary in order. It was never a decision about retrieval. It
was a decision about how to write down a JSON schema, made weeks earlier, for
completely different reasons.

But a fixed cutoff turns any concatenation order into a priority list. The fields at
the front are protected. The fields at the back are optional. So the schema order
silently became the ranking of what matters for search, and it ranked `solution`
fourth.

If I had been asked to rank those fields by search value, `solution` would have been
first or second. I was never asked. The `join` asked instead, and it used the only
order it had.

## Why nothing complained

A cut document still makes a vector. It is still 1024 dimensions. It still has a
similarity score. It still ranks. It is just a vector for the first half of the
document, and nothing says so.

If the embedder threw an error on long input, I would have found this on day one.
Instead the library quietly trims and returns something that looks fine. That is the
right default for a library. It is a trap if your documents are long.

## The fix: stop making one vector per document

The usual answer is to chunk by length. Cut the text into 512-token pieces, embed each
one, store them all. I did something narrower.

My documents already have structure, so instead of chunking by length I embed **by
field**. One vector for `problem`, one for `root_cause`, one for `solution`. Each one
fits under the limit on its own, so nothing is cut and there is no order to fall off
the end of.

Search then runs one KNN query per field and merges those with BM25. The query is
embedded once and reused for all three, so the extra queries cost nothing at embedding
time.

This turned out better than chunking by length, for a reason I did not expect. Because
the fields are separate, they can have **different weights**. A query is always
someone's unsolved problem, so it should look most like a `problem` field and least
like a `solution`. That is now an explicit number I can argue about, instead of a
concatenation order nobody ever looked at.

There is a cost. Three vectors per document instead of one means a bigger index and
more work when a ticket comes in. At this size that is a good trade.

## What to check in yours

Two things, and the first takes twenty minutes.

Measure the token length of your documents against your model's limit. Not the
average. The **share that goes over, and how far the longest ones go**. If a lot of
them are over, you do not have a ranking problem to tune. You have documents that are
half missing from the index, and no reranker can find what was never embedded.

Then look at how you build the text you embed. If it is a concatenation, write down
the order. That order is a priority list whether you meant it to be one or not. The
last field in it is the one you are least likely to retrieve.
