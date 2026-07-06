# Novel Translator

A hybrid Chinese → English web novel translation pipeline. Translates raw
chapters, publishes them as a browsable HTML library, and keeps a glossary
of names/terms and character facts consistent across chapters.

## How it works

Each chapter goes through a few stages:

1. **Number normalization** — Chinese numeral units (万/亿/千) are converted
   to plain Arabic numerals before anything else touches the text.
2. **Local MT draft pass** — [Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)
   (via `transformers`) produces a literal, fully-local translation. No LLM,
   no internet required after the model's first download.
3. **LLM refinement** — a local model running in [LM Studio](https://lmstudio.ai/)
   polishes that draft: fixes grammar and awkward phrasing, enforces
   consistent terminology from the glossary, and leaves numbers and plot
   content untouched. Editing an already-correct draft is a much smaller
   task for the LLM than translating from scratch, which avoids
   "thinking mode" eating the whole token budget on longer chapters.
4. **Entity & fact extraction** — a second LLM call reads the translated
   text and pulls out new named entities and any numeric facts tied to
   them (ages, prices, distances, etc.).
5. **Glossary & continuity check** — new entities are added to the
   glossary (SQLite); new facts are compared against previously stored
   facts for the same entity, and contradictions get logged to
   `flags.log` for manual review.
6. **Chapter title translation** — chapter titles are translated
   separately (and any `第01章`-style numbering prefix is stripped, since
   it's already shown in the page header).
7. **Publish** — the chapter is rendered to `output/chapter_XXX.html`,
   and `output/index.html` (the table of contents) is regenerated from
   the glossary database.

## Requirements

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) running locally with a model loaded
  and its local server started (default: `http://localhost:1234`)

Install dependencies:

```bash
pip install -r requirements.txt
```

The first translation run will download the local MT model
(`Helsinki-NLP/opus-mt-zh-en`, ~300MB) from Hugging Face. After that,
translation works fully offline aside from the LM Studio calls, which
are also local.

## Usage

### GUI

```bash
python app.py
```

Open `http://localhost:8000`. Paste raw Chinese chapter text, set the
chapter number (and optionally a title), and click **Translate & Publish**.
The result is saved into `output/` and shown in the sidebar.

### CLI

```bash
python translate.py raw_chapters/chapter_001.txt 1 "Optional Title"
```

## Project structure

```
pipeline.py     # core translation pipeline (used by both app.py and translate.py)
app.py          # Flask GUI — paste-and-publish dashboard
translate.py    # CLI entry point
glossary.db     # SQLite: entities, facts, and chapter records (gitignored)
output/         # published chapter HTML + index.html
flags.log       # continuity/contradiction flags for review (gitignored)
```

## Configuration

Edit the top of `pipeline.py` to change:

- `LM_STUDIO_URL` / `MODEL_NAME` — which local model LM Studio serves
- `MT_MODEL_NAME` — the local MT model (defaults to `Helsinki-NLP/opus-mt-zh-en`)
- `MT_BATCH_SIZE` — how many lines are translated per MT forward pass

## Publishing as a website

Pushing changes to `output/` on `main` triggers a GitHub Actions workflow
(`.github/workflows/pages.yml`) that deploys the contents of `output/` to
GitHub Pages.

## Notes

- This tool translates existing copyrighted web novel content. Consider
  whether a public repo/site is appropriate for your use case before
  publishing translated chapters widely.
