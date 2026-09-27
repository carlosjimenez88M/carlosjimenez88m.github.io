# The Probability Engine

Personal blog of **Carlos Daniel Jiménez**.

**Music, meaning, and reliable AI.** Research on interpretation and evidence,
with a focused consulting offer for agent memory, retrieval, and evaluation.

🌐 **Live:** https://carlosdanieljimenez.com/

Two primary lines of research and writing:

1. **AI software engineering** — the move from MLOps to LLMOps, production AI on GCP,
   edge inference, and agentic systems as a software-architecture problem.
2. **Narrative arcs in music** — studying how songs and albums develop stories and
   arguments through NLP, embeddings, and LLMs, with attention to the limits of computational interpretation.

Built with [Hugo](https://gohugo.io/) and the [PaperMod](https://github.com/adityatelange/hugo-PaperMod)
theme, deployed to GitHub Pages.

---

## Repository layout

```
.
├── content/                # Markdown source for posts and pages
│   ├── post/               # Essays (one .md per post)
│   ├── about.md            # /about/  (single source — no duplicates)
│   ├── start-here.md       # Editorial entry point
│   ├── work-with-me.md     # Scoped consulting offer
│   ├── experiments.md      # Designs, outputs, and evidence status
│   ├── research-notes.md   # Methods and dated correction log
│   ├── ai-engineering.md   # Hub page (layout: hub)
│   └── music-analysis.md   # Hub page (layout: hub)
├── layouts/                # Theme overrides
│   ├── post/list.html      # Complete chronological writing index
│   ├── index.html          # Minimal homepage with both primary research lines
│   ├── _default/hub.html   # Section hub pages
│   └── partials/           # post_meta, extend_head, extend_footer
├── assets/css/extended/    # custom.css — the theme's design layer
├── static/img/             # Images (served at /img/...)
├── tidytuesday/            # Data-analysis projects behind the music posts
├── themes/PaperMod/        # Theme (only PaperMod is used)
├── hugo.toml               # Site configuration
└── deploy.sh               # Build + publish script
```

> **Note on `static/img/` vs root `img/`:** put new images in `static/img/`.
> The root-level `img/` (and the rest of the compiled HTML at the repo root) is
> generated output — `deploy.sh` copies `public/*` to the root, which is what
> GitHub Pages serves.

---

## Local development

```bash
# Live preview at http://localhost:1313 (includes drafts)
hugo server -D

# Production build into public/
hugo --gc --minify
```

## Publishing

```bash
./deploy.sh "Descriptive commit message"
```

The script builds the site (`hugo --cleanDestinationDir --minify`), copies `public/*`
to the repo root, commits, and pushes to `master`. GitHub Pages serves the result
within a few minutes.

---

## Writing a new post

```bash
hugo new post/my-new-post.md
```

Front matter used across posts:

```yaml
---
author: Carlos Daniel Jiménez
date: 2026-06-16
title: "..."
categories: ["Music Analysis", "LLMs"]
tags: ["nlp", "embeddings", "..."]
---
```

Music posts pair with an analysis folder under `tidytuesday/` (Python notebooks +
scripts that produce the figures the post embeds from `/tidytuesday/<slug>/`).

---

## Secrets

API keys and credentials live in a local `.env` that is **git-ignored** — see
`.gitignore`. Never commit `.env`, `*.key`, `*.pem`, or service-account JSON.

---

## Editorial and distribution workflow

- [LinkedIn and X plan](docs/visibilidad-linkedin-x.md): four weeks of posts, profile copy, and UTM links.
- [Measurement and publishing](docs/medicion-y-publicacion.md): optional GA4 setup, event meanings, and deployment boundaries.
- Run `python3 scripts/audit_editorial_exports.py` to regenerate the file-inspection provenance summary; this makes no model calls.
- Run `python3 scripts/create_social_cards.py` with Pillow to regenerate sharing images. The PNGs are committed under `static/img/social/`.

Long essays can include a reading guide in their front matter:

```yaml
readerGuide:
  summary: "The question, what the essay examines, and what the reader can learn."
  scope: "The corpus, evidence status, and limits of the conclusion."
  resources:
    - label: "Inspect the experiment"
      url: "/experiments/"
```

Keep corrections dated with `lastmod` and a visible `editorialNote`; record substantive changes in `content/research-notes.md`. Do not describe a stored-output inspection as a new model run or human validation.

## Contact

- **Email:** danieljimenez88m@gmail.com
- **GitHub:** [@carlosjimenez88M](https://github.com/carlosjimenez88M)
- **LinkedIn:** [djimenezm](https://www.linkedin.com/in/djimenezm)
- **X:** [@DanielJimenezM9](https://x.com/DanielJimenezM9)

Content © Carlos Daniel Jiménez.
