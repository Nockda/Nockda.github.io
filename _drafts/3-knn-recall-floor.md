---
layout: post
project: true
title: Two max() calls gave my KNN the worst possible recall
subtitle: num_candidates, k, and a one-line fix
tags: [rag, python]
comments: true
share-description: "Elasticsearch KNN takes k and num_candidates. The ratio between them is the accuracy dial, and two ordinary-looking max() calls flattened mine to 1.0 — the lowest the engine allows."
---

<!--
  DRAFT 3 of 4. SANITISED: no employer, no product names, no ticket data.
  Short on purpose — one idea, one fix.
-->

Elasticsearch KNN takes two numbers that are easy to mix up:

- **`k`** — how many results you want.
- **`num_candidates`** — how many nodes HNSW looks at before picking those `k`.

HNSW is an *approximate* search. It does not compare your query to every vector. It
walks a graph and keeps the best candidates it finds on the way. `num_candidates` is
how many it keeps while walking.

So the **ratio** between the two is the accuracy dial. The closer `num_candidates` is
to `k`, the more the search misses. When they are equal, you have told the engine to
keep whatever it bumps into and look no further. That is the lowest setting
Elasticsearch allows, and the worst recall you can get.

## Two floors that ate the multipliers

Here is the code. I think most reviewers would let it through:

```python
"num_candidates": max(top_k * 10, 50),
"k":              max(top_k * 5,  50),
```

Ten times over-fetch on candidates, five times on `k`. The multipliers look sensible.
And `max(x, floor)` is such a common safety pattern that nothing looks wrong.

The problem shows up at small `top_k`. At the default of 5:

- `top_k * 10` → 50. The floor is 50. Same number.
- `top_k * 5` → 25. The floor is 50. Pushed **up** to 50.

Both end up at 50. The multipliers never do anything, and the ratio drops to 1.0.

| Path | `top_k` | `num_candidates` | `k` | ratio |
| :--- | :--: | :--: | :--: | :--: |
| Search | 5 | 50 | 50 | **1.0×** |
| Chatbot | 3 | 50 | 50 | **1.0×** |
| top_k = 10 | 10 | 100 | 50 | 2.0× |

<svg viewBox="0 0 720 250" role="img" width="100%"
     aria-label="At top_k of 5, num_candidates resolves to max of 50 and 50, and k resolves to max of 25 and 50. Both are clamped to the floor of 50, so the ratio collapses from the intended 2 to 1. Deriving num_candidates from the pool instead keeps the ratio at 5."
     style="max-width:720px;margin:2rem auto;display:block;font-family:'Open Sans',Helvetica,Arial,sans-serif">

  <text x="20" y="22" font-size="11.5" letter-spacing="1.2" fill="var(--ink-soft)">BEFORE — top_k = 5</text>

  <!-- num_candidates path -->
  <text x="20" y="58" font-size="12.5" font-family="ui-monospace,Menlo,monospace" fill="var(--ink-soft)">num_candidates</text>
  <text x="150" y="58" font-size="12.5" font-family="ui-monospace,Menlo,monospace" fill="var(--ink)">max(5 &#215; 10, 50)</text>
  <text x="290" y="58" font-size="12.5" fill="var(--ink-soft)">&#8594;</text>
  <text x="315" y="58" font-size="12.5" font-family="ui-monospace,Menlo,monospace" fill="var(--ink-soft)">max(50, 50)</text>
  <text x="425" y="58" font-size="12.5" fill="var(--ink-soft)">&#8594;</text>
  <rect x="450" y="43" width="52" height="22" rx="3" fill="var(--surface)" stroke="var(--border)"/>
  <text x="476" y="59" text-anchor="middle" font-size="13" font-weight="700" font-family="ui-monospace,Menlo,monospace" fill="var(--ink)">50</text>

  <!-- k path -->
  <text x="20" y="96" font-size="12.5" font-family="ui-monospace,Menlo,monospace" fill="var(--ink-soft)">k</text>
  <text x="150" y="96" font-size="12.5" font-family="ui-monospace,Menlo,monospace" fill="var(--ink)">max(5 &#215; 5, 50)</text>
  <text x="290" y="96" font-size="12.5" fill="var(--ink-soft)">&#8594;</text>
  <text x="315" y="96" font-size="12.5" font-family="ui-monospace,Menlo,monospace" fill="var(--ink-soft)">max(<tspan font-weight="700" fill="var(--ink)">25</tspan>, 50)</text>
  <text x="425" y="96" font-size="12.5" fill="var(--ink-soft)">&#8594;</text>
  <rect x="450" y="81" width="52" height="22" rx="3" fill="var(--surface)" stroke="var(--border)"/>
  <text x="476" y="97" text-anchor="middle" font-size="13" font-weight="700" font-family="ui-monospace,Menlo,monospace" fill="var(--ink)">50</text>

  <!-- collapse brace -->
  <path d="M512 50 h14 v46 h-14" fill="none" stroke="var(--ink-soft)" stroke-width="1.5"/>
  <text x="538" y="66" font-size="12.5" fill="var(--ink)" font-weight="700">ratio 1.0&#215;</text>
  <text x="538" y="84" font-size="11.5" fill="var(--ink-soft)">the engine floor</text>

  <line x1="20" y1="130" x2="700" y2="130" stroke="var(--border)" stroke-width="1"/>

  <text x="20" y="158" font-size="11.5" letter-spacing="1.2" fill="var(--ink-soft)">AFTER — derived from the pool</text>

  <text x="20" y="194" font-size="12.5" font-family="ui-monospace,Menlo,monospace" fill="var(--ink-soft)">pool</text>
  <text x="150" y="194" font-size="12.5" font-family="ui-monospace,Menlo,monospace" fill="var(--ink)">max(5 &#215; 5, 50)</text>
  <text x="290" y="194" font-size="12.5" fill="var(--ink-soft)">&#8594;</text>
  <rect x="315" y="179" width="52" height="22" rx="3" fill="var(--surface)" stroke="var(--border)"/>
  <text x="341" y="195" text-anchor="middle" font-size="13" font-weight="700" font-family="ui-monospace,Menlo,monospace" fill="var(--ink)">50</text>

  <text x="20" y="228" font-size="12.5" font-family="ui-monospace,Menlo,monospace" fill="var(--ink-soft)">num_candidates</text>
  <text x="150" y="228" font-size="12.5" font-family="ui-monospace,Menlo,monospace" fill="var(--ink)">pool &#215; 5</text>
  <text x="290" y="228" font-size="12.5" fill="var(--ink-soft)">&#8594;</text>
  <rect x="315" y="213" width="62" height="22" rx="3" fill="var(--accent)"/>
  <text x="346" y="229" text-anchor="middle" font-size="13" font-weight="700" font-family="ui-monospace,Menlo,monospace" fill="var(--page)">250</text>

  <path d="M387 186 h14 v46 h-14" fill="none" stroke="var(--accent)" stroke-width="1.5"/>
  <text x="413" y="202" font-size="12.5" fill="var(--ink)" font-weight="700">ratio 5.0&#215;</text>
  <text x="413" y="220" font-size="11.5" fill="var(--ink-soft)">cannot collapse</text>
</svg>

The over-fetch only starts working at `top_k = 10` or higher. Nobody was using that.
The two busiest paths ran at the minimum.

Here is the general lesson. When you clamp two related values on their own, **the
relationship between them can disappear** — and here that relationship was the whole
point. A floor on one number is safe. A floor on both ends of a ratio is a bug waiting
for a small input.

## The fix

Work out the candidate count from the pool instead of clamping it separately:

```python
def _pool(top_k: int) -> int:
    # every ranking going into the merge must be the same length,
    # or the merge tilts toward whichever leg returned more
    return max(top_k * 5, 50)

def _num_candidates(pool: int) -> int:
    override = os.getenv("ES_KNN_NUM_CANDIDATES", "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)
    return pool * 5
```

Now the 5× ratio holds whatever the pool is, and it cannot collapse. The environment
variable is there so the value can be tuned later against a test set, instead of
staying a number that just sounds right.

There is a cost. HNSW now visits five times as many nodes per query. At this size the
extra latency was too small to bother measuring, and the recall was worth it. At ten
million vectors the answer would be different. `num_candidates` is a dial, not a
constant, and the right value depends on how much latency you can spend.

This was the cheapest accuracy fix in the system. One line. No migration, no
re-indexing.

## Worth checking in yours

If you run approximate vector search, print the two numbers your queries actually end
up with at your *default* settings. Not the ones in the config file. The ones left
after every `max()`, `min()` and default has been applied.

They are often not what you think.

<!--
  TODO before publishing:
    - [ ] read aloud
    - [ ] old values verified against git (commit 1ddab33^): max(top_k*10,50) / max(top_k*5,50) — confirmed
    - [ ] move to _posts/YYYY-MM-DD-knn-recall-floor.md
-->
