# Improvement plan — nockda.github.io

Working document for turning this blog into a portfolio that supports the CV.
Derived from the audit of **2 September 2026**. Tick items as they land.

> **The premise.** The site is well built; the build is green and the technical SEO
> is correct. The problem is that the CV claims production RAG and agentic LLM work
> and the blog contains eight posts of 2023 MSc coursework. Everything below is
> ordered so the cheap credibility fixes land first and the one expensive item —
> writing about current work — is not blocked behind them.

**Success condition:** a hiring manager who reads the CV, clicks through, and spends
ninety seconds here comes away with the CV *confirmed* rather than doubted.

---

## Phase 0 — Tonight (~1 hour, no writing required)

These are typing tasks. None of them need a draft, a decision, or a weekend.

### 0.1 Stop the repo advertising the template — P4

The repository behind the site still carries Beautiful Jekyll's marketing copy.
A recruiter arriving via GitHub is sent to someone else's website.

- [ ] Set description, homepage and topics:

```bash
gh repo edit Nockda/Nockda.github.io \
  --description "Personal blog and project portfolio — AI engineering, LLM systems, data engineering." \
  --homepage "https://nockda.github.io" \
  --add-topic jekyll --add-topic blog --add-topic portfolio \
  --add-topic machine-learning --add-topic data-engineering
```

**Done when:** `gh repo view Nockda/Nockda.github.io` shows your description, not
`✨ Build a beautiful and simple website in literally minutes`.

### 0.2 Fix the eleven spelling errors — P8

All of these are in headings or alt text, which is exactly where a skimming reader
looks. Exact locations:

| File | Line | Wrong | Right |
| :--- | ---: | :--- | :--- |
| `_posts/2023-11-29-customer.md` | 60 | Rediness | Readiness |
| `_posts/2023-11-29-customer.md` | 89 | Devide | Divide |
| `_posts/2023-11-29-customer.md` | 302 | Huristic | Heuristic |
| `_posts/2023-06-12-Semiconductor-Manufacturing-Analysis.md` | 50 | Rediness | Readiness |
| `_posts/2023-06-12-Semiconductor-Manufacturing-Analysis.md` | 9, 17 | lightbgm / lightBGM | LightGBM |
| `_posts/2023-06-20-Model-for-Fraud-Detection-System.md` | 47 | Rediness | Readiness |
| `_posts/2023-12-10-Credit-Card-Fraud-Detection-Model.md` | 40, 58 | anormaly | anomaly |
| `_posts/2023-06-05-Classification.md` | 9 | Maching learning | machine learning |
| `_posts/2023-06-01-Airbnb-Clone-Coding.md` | 74 | iamge | image |

- [ ] Apply the replacements
- [ ] Capitalise the Airbnb title: `title: airbnb clone coding` → `title: Airbnb Clone Coding`

**Done when:** this returns nothing:

```bash
grep -rniE '\b(rediness|devide|huristic|iamge|anormaly|maching|lightbgm)\b' _posts/
```

### 0.3 Separate the two fraud-detection posts — P7

They currently share the identical subtitle `AutoEncoder & IF & LOF`, and the
December post does not use autoencoders at all — it compares logistic regression,
KNN, LDA, SVM and random forest.

- [ ] `2023-06-20-Model-for-Fraud-Detection-System.md` — keep `AutoEncoder & IF & LOF`
      (accurate for this one)
- [ ] `2023-12-10-Credit-Card-Fraud-Detection-Model.md` — replace with something true,
      e.g. `Five classifiers on a 0.17% positive class`

**Done when:** the Projects page no longer shows two cards that read as the same project.

### 0.4 Favicon and dead social tags — P5

- [ ] Add `favicon.ico` (and a 180×180 `apple-touch-icon.png`) at the repo root —
      currently `https://nockda.github.io/favicon.ico` returns **404**
- [ ] The theme emits `<meta name="twitter:site" content="@">` because no Twitter
      handle is set. Either add the handle or delete those two lines from
      `_includes/head.html`

### 0.5 Turn on analytics — P10

Nothing is configured, so there is currently no way to know whether recruiters reach
the site at all, which post they open, or whether the CV link is ever clicked.

- [ ] Create a GA4 property, then uncomment and fill the stub already in `_config.yml`:

```yaml
gtag: "G-XXXXXXXXXX"
```

**Done when:** a visit from your own phone shows up in GA4 realtime.

---

## Phase 1 — This month (2 evenings + 1 weekend)

### 1.1 Publish one post about current LLM work — P1, P2 ⭐

**This is the whole plan in one task.** Everything else is maintenance; this is the
item that changes what a reader concludes. It resets the 32-month timeline *and*
closes the CV↔blog gap in a single move.

Nothing confidential is required. What is being hired is your **reasoning**, and
reasoning is not proprietary. Candidate topics, all writable from public knowledge
plus your own judgement:

- **Why hybrid retrieval beat pure dense search for us** — BM25 + vector KNN in
  Elasticsearch, why exact-match recall mattered for support tickets, what dense-only
  kept missing. Directly mirrors the CV bullet.
- **Serving two LLM services on one GPU** — memory budgeting, 4-bit quantisation,
  what you gave up in exchange for throughput. Concrete, unusual, hard to fake.
- **What precision@k stopped telling us** — building an evaluation loop, why
  retrieval metrics diverged from user-perceived answer quality, what you added.
- **Giving an agent controlled access to CRM data through MCP** — the permission
  model, the tool boundary, what you refused to expose. Very few people have written
  this from production experience.

Rules for the post:

- [ ] Lead with the problem and the outcome; keep the tutorial parts short
- [ ] Include one decision you got wrong and changed — this is the highest-credibility
      paragraph you can write
- [ ] Generalise employer specifics; no customer data, no internal identifiers
- [ ] Front matter: `project: true`, real `tags`, a real `share-description`

**Done when:** the newest post on the homepage is about LLM systems, not Apache Spark.

### 1.1a How to write it — voice

Two things at once: **plain English, exact technical terms.** Simple English is about
the sentences, not the vocabulary. Never water down `BM25`, `precision@k`,
`4-bit quantisation` or `hybrid retrieval` — those words are the evidence. Water down
everything around them.

The reader is a hiring manager or a senior engineer skimming on a Tuesday afternoon.
They want to know what you decided and why. They do not want an essay.

**Write it like a person, not like a model.**

The tell is not vocabulary, it is rhythm. AI prose keeps every sentence the same
length, opens every section with a summary of what the section will say, and closes
with a moral. People don't do that. People start in the middle, give a number, and
move on.

Specific things to cut:

- **The textbook opener.** Your credit-card post begins: *"In today's society, the
  frequency of credit card transactions has increased significantly."* That sentence
  tells the reader nothing they didn't know, and it is exactly the shape an LLM
  produces. Start with the actual problem: *"Only 0.17% of the transactions in this
  dataset are fraud. That breaks accuracy as a metric, so I had to pick a different
  one."*
- **Padding verbs.** *"Credit card companies bear the responsibility to protect the
  customers' assets"* → *"Credit card companies have to protect their customers'
  money."*
- **Announcing structure.** "In this section, we will explore…", "Let's dive in",
  "It is worth noting that". Delete the sentence; keep what came after it.
- **Words you would not say out loud.** delve, leverage (as a verb), utilise,
  robust, seamless, comprehensive, furthermore, moreover. Use *use*, *strong*,
  *full*, *also*.
- **Three-item lists everywhere.** If you have two reasons, give two.
- **The tidy conclusion.** "In conclusion, this project demonstrates…" Stop when you
  run out of things to say. Or end on what you would do next time.

Specific things to keep:

- **Short sentences.** One idea each. Fifteen to twenty words is plenty. Then a long
  one, when you need it — the variation is what makes it sound spoken.
- **"I", not "we"** when you did it alone. The Gerald McDonald RAG service was you,
  on your own. Say so: *"I was the only engineer on it."*
- **Real numbers over adjectives.** Not *"significantly faster"* but *"resolution
  time dropped about 40%"*. Not *"a large dataset"* but *"284,807 rows"*.
- **The messy part.** *"My first version used dense retrieval only. It kept missing
  tickets where the user quoted an exact error code, so I added BM25 back in."*
  That paragraph does more for you than three paragraphs of architecture.
- **Plain contractions.** "don't", "didn't", "it's". Formal English reads as
  translated; contractions read as spoken.

Practical trick: write each paragraph as if you were explaining it to a colleague at
your desk, then read it back aloud. Anything you would not say out loud, rewrite.
Long, formal English is not better English — for this audience it is worse, because
it hides the reasoning you are being hired for.

**Self-check before publishing:**

- [ ] Read it aloud. Any sentence you stumble on gets shortened or split
- [ ] First sentence names a real problem, not a trend
- [ ] At least one number in the first paragraph
- [ ] At least one thing you got wrong
- [ ] No sentence longer than about 25 words unless it earns it
- [ ] Every technical term is the exact one, spelled the way the field spells it
      (LightGBM, not lightBGM — see 0.2)

### 1.2 Make every project verifiable — P3

Six of eight posts link to no repository. For an engineering reader, a project
description without code is a claim, not evidence.

- [ ] Add a repo link directly under the front matter of each project post
- [ ] Where code cannot be shared, say so explicitly — *"Built at work; code is
      private. The approach is described below."* An acknowledged private repo reads
      far better than silence
- [ ] Link the existing public `dissertation` repo from the relevant post
- [ ] Consider whether `Portfolio-website` (React, last touched 2025-03) should be
      archived — a second, staler personal site splits the reader's attention

### 1.3 Write real search descriptions — P5

The theme uses `subtitle` as the meta description, so the 3,345-word classifier
comparison currently advertises itself to Google as *"AutoEncoder & IF & LOF"*.
Four posts have an empty subtitle and fall back to a truncated excerpt.

- [x] Add `share-description` to all eight posts — one sentence naming the problem
      and the result, e.g.

```yaml
share-description: "Comparing five classifiers on a credit-card dataset where only 0.17% of transactions are fraudulent, and why accuracy is the wrong metric."
```

### 1.4 Compress the images — P6

| File | Now |
| :--- | ---: |
| `assets/img/cancer_classification/cancer_cover.png` | 1,950,778 B |
| `assets/img/spark/spark_cover.png` | ~1.8 MB |
| `assets/img/semiconductor/cover.png` | 1,564,911 B |
| `assets/img/` total | ~26 MB |

- [x] Resize covers to 1600 px wide and re-encode. Keeping the filenames means no
      markup changes are needed:

```bash
# needs: brew install imagemagick
find assets/img -name '*.png' -size +300k -exec \
  magick mogrify -resize '1600>' -strip -quality 82 {} \;
```

- [ ] Consider WebP for the largest covers (~90% reduction, no visible difference)

**Target:** no single image over 300 KB; `assets/img` under 5 MB.

### 1.5 Collapse the tag system — P9

34 distinct tags across 8 posts; 33 are used exactly once. Only `python` repeats.
Every other tag leads to a page with one result, so the Tags link in the navigation
promises browsing it cannot deliver.

Proposed vocabulary — six tags, each used more than once:

| Tag | Posts |
| :--- | :--- |
| `llm` | new posts |
| `rag` | new posts |
| `machine-learning` | Classification, Semiconductor, Fraud Detection, Credit Card |
| `data-engineering` | Big Data, Spark Intro, Semiconductor |
| `analytics` | Customer RFM, Semiconductor |
| `python` | all except Airbnb |
| `web` | Airbnb |

- [x] Rewrite the `tags:` line in all eight posts against this list
- [x] Delete tags that name a single library (`lightbgm`, `LocalOutlierFactor`,
      `IsolationForest`, `KNN`, `LDA`, `SVM`) — those belong in the body text, not
      in navigation

---

## Phase 2 — Keep it alive (one post per quarter)

Four posts a year beats eight posts in one summer. The cadence is the signal.

- [ ] **Second LLM post** so the first does not read as a one-off
- [ ] **Reorder the Projects page** to put current work above 2023 coursework —
      the page sorts by date, so this happens automatically once new posts exist
- [ ] **Review analytics after each application round** and write toward whatever
      actually gets opened
- [ ] **Keep the CV in sync** — `assets/cv/HyunSuk_Lee_CV.pdf` is now self-hosted,
      so it cannot rot the way the Google Drive link did; regenerate it whenever the
      CV changes

### Optional, once the content gap is closed

- [ ] Enable comments — install the [Utterances app](https://github.com/apps/utterances),
      turn on Issues, then uncomment the `utterances` block in `_config.yml`
- [ ] Add `alt` text to the 16 `<img>` tags that lack it, and replace the placeholder
      markdown alts (`![1]`, `![2]`) with descriptions
- [ ] Fix the heading structure — posts currently jump `h2` → `h4` with no `h3`

---

## Do not touch

The temptation after an audit is to rebuild things that are fine. These are working:

- **The build** — CI green, deploys cleanly, Projects page generates itself from
  `project: true` front matter
- **Technical SEO** — sitemap, canonical URLs, OG images and RSS all emit correctly.
  Only the copy inside them is weak, and 1.3 fixes that
- **The writing** — clear, unpadded technical English that explains its reasoning.
  This is the hard part and it is already there
- **Design** — readable typography, working dark mode, sensible navigation. Design is
  not what is holding this site back

---

## Priority reference

| ID | Finding | Severity | Phase |
| :-- | :--- | :--- | :--- |
| P1 | Blog evidences a job you no longer apply for | Critical | 1.1 |
| P2 | 32-month silence covering both current roles | Critical | 1.1 |
| P3 | 6 of 8 posts link to no code | High | 1.2 |
| P4 | Repo advertises the template | High | 0.1 |
| P5 | Search snippets and favicon | Medium | 0.4, 1.3 |
| P6 | Multi-megabyte images | Medium | 1.4 |
| P7 | Duplicate fraud-post subtitles | Medium | 0.3 |
| P8 | Spelling errors in headings | Medium | 0.2 |
| P9 | 34 tags, 33 used once | Low | 1.5 |
| P10 | No analytics | Low | 0.5 |

Full audit with evidence and the timeline figure:
<https://claude.ai/code/artifact/027d4641-e9e1-4a4f-8d3e-8ca0a836bf39>
