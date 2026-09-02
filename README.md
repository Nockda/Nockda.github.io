# nockda.github.io

Personal blog and project portfolio of HyunSuk Lee — data science, data
engineering and machine learning notes.

Live at **https://nockda.github.io**.

Built with [Beautiful Jekyll](https://beautifuljekyll.com) (v6.0.1) and hosted
on GitHub Pages.

## Running locally

```bash
bundle install
bundle exec jekyll serve --livereload
```

The site is then at <http://localhost:4000>.

## Adding a post

Create `_posts/YYYY-MM-DD-title.md` with front matter:

```yaml
---
layout: post
title: Post title
subtitle: Optional one-liner
cover-img: /assets/img/<folder>/cover.png
thumbnail-img: /assets/img/<folder>/logo.png
share-img: /assets/img/<folder>/logo.png
tags: [python, spark]
comments: true
---
```

Add `project: true` to also list it on the [Projects](https://nockda.github.io/projects)
page. The navigation bar is generated from `_config.yml` and the Projects page
builds itself from front matter — neither needs editing when a post is added.

## Layout

| Path | Purpose |
| --- | --- |
| `_config.yml` | Site settings, navigation, colours |
| `_posts/` | Blog posts and project write-ups |
| `_layouts/`, `_includes/` | Vendored Beautiful Jekyll templates |
| `assets/css/custom.css` | Custom styling and the dark theme |
| `assets/img/` | Post images, one folder per post |
| `aboutme.md`, `projects.html`, `tags.html` | Standalone pages |
