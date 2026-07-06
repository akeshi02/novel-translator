"""
pipeline.py — core translation logic, shared by translate.py (CLI) and
app.py (GUI). Keeping this separate means the GUI and the command-line
tool never drift out of sync with each other.
"""

import os
import re
import json
import sqlite3
import requests
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "qwen/qwen3-8b"   # matches the name shown in LM Studio's server tab
TEMPERATURE = 0.3

# Local MT model for the initial draft pass. Helsinki-NLP/opus-mt-zh-en is
# free, fully local/offline after the first download, and downloads via
# huggingface_hub, which does chunked + resumable downloads — much more
# reliable over a flaky connection than argostranslate's raw download,
# which is what kept giving IncompleteRead errors.
MT_MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"
MT_BATCH_SIZE = 16  # lines translated per forward pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "glossary.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FLAGS_LOG = os.path.join(BASE_DIR, "flags.log")


# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS glossary (
            entity TEXT PRIMARY KEY,
            translation TEXT,
            entity_type TEXT,
            first_chapter INTEGER
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity TEXT,
            value REAL,
            unit TEXT,
            chapter INTEGER,
            raw_context TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chapters (
            chapter INTEGER PRIMARY KEY,
            title TEXT,
            translated_at TEXT
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# NUMBER NORMALIZATION
# ---------------------------------------------------------------------------
def convert_chinese_numbers(text):
    def repl_wan(m):
        return str(int(float(m.group(1)) * 10_000))

    def repl_yi(m):
        return str(int(float(m.group(1)) * 100_000_000))

    def repl_qian(m):
        return str(int(float(m.group(1)) * 1_000))

    text = re.sub(r'(\d+(?:\.\d+)?)亿', repl_yi, text)
    text = re.sub(r'(\d+(?:\.\d+)?)万', repl_wan, text)
    text = re.sub(r'(\d+(?:\.\d+)?)千', repl_qian, text)
    return text


# ---------------------------------------------------------------------------
# LOCAL MT (Helsinki-NLP/opus-mt-zh-en) FOR THE INITIAL (DRAFT) PASS
# ---------------------------------------------------------------------------
# The LLM used to be responsible for translating from scratch, which meant
# every chapter needed a big, "creative" completion — exactly the kind of
# request that's most vulnerable to the thinking-mode token-budget failures
# we kept running into. Now this local MarianMT model (fully free, fully
# offline after the first download, no LLM involved) does the literal first
# pass, and the LLM's job shrinks down to editing an already-correct-in-
# substance draft: fixing grammar/flow and enforcing the glossary. That's a
# much smaller, more mechanical task for the LLM.
_mt_tokenizer = None
_mt_model = None


def ensure_local_mt_model():
    """
    Loads the Helsinki-NLP/opus-mt-zh-en tokenizer + model, downloading it
    from Hugging Face on first use (cached under ~/.cache/huggingface after
    that, so every run after the first is fully offline). Safe to call
    repeatedly — it's a no-op once loaded.
    """
    global _mt_tokenizer, _mt_model
    if _mt_model is not None:
        return

    try:
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError:
        raise RuntimeError(
            "The 'transformers' package isn't installed. Run: "
            "pip install transformers torch sentencepiece"
        )

    try:
        _mt_tokenizer = MarianTokenizer.from_pretrained(MT_MODEL_NAME)
        _mt_model = MarianMTModel.from_pretrained(MT_MODEL_NAME)
    except Exception as e:
        _mt_tokenizer = None
        _mt_model = None
        raise RuntimeError(
            f"Could not load/download the local MT model {MT_MODEL_NAME!r}: "
            f"{e}. If you're offline, this model needs to be downloaded at "
            "least once — check your connection and try again. Hugging "
            "Face downloads resume automatically if interrupted, so just "
            "rerunning often fixes a partial/failed download."
        )


def local_mt_translate(text):
    """
    Runs the local MT pass over `text`, batching non-empty lines together
    for speed, and reassembles the result preserving the original line
    breaks (including blank lines, which mark paragraph breaks downstream).
    """
    ensure_local_mt_model()

    lines = text.split("\n")
    non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
    translated = [""] * len(lines)

    for start in range(0, len(non_empty_indices), MT_BATCH_SIZE):
        batch_indices = non_empty_indices[start:start + MT_BATCH_SIZE]
        batch_lines = [lines[i] for i in batch_indices]

        tokens = _mt_tokenizer(
            batch_lines, return_tensors="pt", padding=True, truncation=True
        )
        generated = _mt_model.generate(**tokens, max_new_tokens=512)
        decoded = _mt_tokenizer.batch_decode(generated, skip_special_tokens=True)

        for i, text_out in zip(batch_indices, decoded):
            translated[i] = text_out

    return "\n".join(translated)


# ---------------------------------------------------------------------------
# LM STUDIO CALL
# ---------------------------------------------------------------------------
def call_lm_studio(system_prompt, user_prompt, temperature=TEMPERATURE, max_tokens=8000):
    # /no_think must be on the USER turn for Qwen's template to honor it —
    # putting it in the system prompt (as an earlier version of this file
    # did) is silently ignored by the chat template.
    user_prompt_no_think = user_prompt + "\n/no_think"

    resp = requests.post(
        LM_STUDIO_URL,
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_no_think},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Belt-and-suspenders: also disable thinking via the
            # chat-template kwarg some backends (vLLM/llama.cpp servers)
            # respect directly, in case /no_think isn't picked up.
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=900,
    )
    resp.raise_for_status()
    message = resp.json()["choices"][0]["message"]
    content = (message.get("content") or "").strip()

    # Some Qwen3-family responses put thinking in a separate
    # "reasoning_content" field even when content is empty. If content
    # came back empty, that's a sign thinking mode ate the whole budget —
    # surface this clearly instead of silently returning "".
    if not content:
        reasoning_preview = (message.get("reasoning_content") or "")[:200]
        raise RuntimeError(
            "Model returned no usable content — thinking mode likely "
            "consumed the entire token budget before producing an answer. "
            f"Reasoning preview: {reasoning_preview!r}. "
            "Try increasing max_tokens further, or confirm 'Enable "
            "Thinking' is off in LM Studio's model settings."
        )

    # Strip <think>...</think> blocks defensively, in case they end up
    # inline in content rather than in a separate reasoning field.
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


# ---------------------------------------------------------------------------
# GLOSSARY
# ---------------------------------------------------------------------------
def get_glossary_text(conn):
    c = conn.cursor()
    c.execute("SELECT entity, translation, entity_type FROM glossary")
    rows = c.fetchall()
    if not rows:
        return "(no glossary entries yet — this may be an early chapter)"
    return "\n".join(
        f"- {ent} ({etype}) → always translate as \"{trans}\""
        for ent, trans, etype in rows
    )


def update_glossary(conn, entities, chapter_num):
    c = conn.cursor()
    for ent in entities:
        name = ent.get("name", "").strip()
        etype = ent.get("type", "unknown")
        if not name:
            continue
        c.execute("SELECT entity FROM glossary WHERE entity = ?", (name,))
        if c.fetchone() is None:
            c.execute(
                "INSERT INTO glossary (entity, translation, entity_type, first_chapter) "
                "VALUES (?, ?, ?, ?)",
                (name, name, etype, chapter_num),
            )
    conn.commit()


# ---------------------------------------------------------------------------
# TRANSLATION
# ---------------------------------------------------------------------------
def translate_chapter(text, glossary_text):
    """
    Two-stage translation:
      1. argostranslate produces a fast, fully-local literal draft — no
         LLM involved, so no thinking-mode token budget to burn.
      2. The LLM refines that draft for grammar, flow, and glossary
         consistency. Editing an already-correct-in-substance draft is a
         much smaller, more mechanical task than translating a chapter
         from scratch, which keeps the LLM prompt short and the failure
         mode we were fighting (empty completions from thinking mode
         eating the whole budget) much less likely.
    """
    try:
        draft = local_mt_translate(text)
    except Exception as e:
        # Fall back to the old behavior (LLM translates from scratch)
        # rather than failing the whole chapter if the local MT model
        # isn't available.
        print(f"⚠ Local MT model unavailable ({e}); falling back to LLM-only translation.")
        system_prompt = (
            "You are a professional Chinese-to-English web novel translator. "
            "Translate naturally and fluently, but preserve all numeric values "
            "EXACTLY as given in the source text — do not round, estimate, or "
            "reinterpret numbers. All Chinese numeral units have already been "
            "converted to plain Arabic numerals in the source; keep them "
            "unchanged in your translation. Use the following glossary for "
            "names and recurring terms — always translate these consistently:\n\n"
            f"{glossary_text}\n\n"
            "Return ONLY the translated chapter text. No preamble, no notes, "
            "no markdown formatting, just the translated prose."
        )
        return call_lm_studio(system_prompt, text)

    system_prompt = (
        "You are a professional editor polishing a rough machine-translated "
        "draft of a Chinese web novel chapter into natural, fluent English. "
        "The draft below was produced by a local MT engine: the meaning and "
        "all numeric values are already correct, but the phrasing may be "
        "literal, awkward, or grammatically rough, and terminology may be "
        "inconsistent. Your job:\n"
        "1. Rewrite awkward or unnatural sentences so they read fluently.\n"
        "2. Fix grammar and punctuation.\n"
        "3. Enforce consistent terminology using the glossary below — "
        "replace any name or term that doesn't match it.\n"
        "4. Preserve all numeric values EXACTLY as given — do not round, "
        "estimate, or change them.\n"
        "5. Do not add, omit, or reinterpret plot content.\n\n"
        f"Glossary:\n{glossary_text}\n\n"
        "Return ONLY the polished chapter text. No preamble, no notes, "
        "no markdown formatting, just the final prose."
    )
    return call_lm_studio(system_prompt, draft)


def extract_facts_and_entities(translated_text):
    system_prompt = (
        "You analyze translated web novel text and extract structured data. "
        "Return ONLY valid JSON, nothing else — no preamble, no markdown "
        "fences. Use exactly this schema:\n"
        '{\n'
        '  "entities": [{"name": "...", "type": "person|place|org|item"}],\n'
        '  "facts": [{"entity": "...", "value": 0, "unit": "yuan|years|km|..."}]\n'
        '}\n'
        "Only include NEW named entities that appear in this chapter. Only "
        "include facts that are specific numeric statements tied to a "
        "specific named entity (e.g. a price for a named item, an age for "
        "a named character) — skip vague numbers with no clear entity."
    )
    try:
        raw = call_lm_studio(system_prompt, translated_text, temperature=0.1, max_tokens=2000)
    except Exception as e:
        print(f"⚠ Fact/entity extraction failed, publishing translation without it: {e}")
        return {"entities": [], "facts": []}

    cleaned = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"entities": [], "facts": []}


def check_and_store_facts(conn, facts, chapter_num, raw_context=""):
    c = conn.cursor()
    flags = []
    for f in facts:
        entity = f.get("entity", "").strip()
        value = f.get("value")
        unit = f.get("unit", "")
        if not entity or value is None:
            continue
        c.execute(
            "SELECT value, unit, chapter FROM facts WHERE entity = ? AND unit = ? "
            "ORDER BY chapter DESC LIMIT 1",
            (entity, unit),
        )
        prev = c.fetchone()
        if prev is not None:
            prev_value, prev_unit, prev_chapter = prev
            if float(prev_value) != float(value):
                flags.append(
                    f"Chapter {chapter_num}: \"{entity}\" changed from "
                    f"{prev_value} {prev_unit} (ch.{prev_chapter}) to "
                    f"{value} {unit} (ch.{chapter_num}). Review needed."
                )
        c.execute(
            "INSERT INTO facts (entity, value, unit, chapter, raw_context) "
            "VALUES (?, ?, ?, ?, ?)",
            (entity, value, unit, chapter_num, raw_context[:200]),
        )
    conn.commit()
    return flags


def log_flags(flags):
    if not flags:
        return
    with open(FLAGS_LOG, "a", encoding="utf-8") as f:
        for flag in flags:
            timestamp = datetime.now().isoformat(timespec="seconds")
            f.write(f"[{timestamp}] {flag}\n")


# ---------------------------------------------------------------------------
# CHAPTER TITLE TRANSLATION
# ---------------------------------------------------------------------------
_CHINESE_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')


def contains_chinese(text):
    return bool(text) and bool(_CHINESE_CHAR_RE.search(text))


def translate_title(raw_title):
    """
    Translates a chapter title (e.g. "第01章 這糟老頭子可壞了") into a short,
    clean English title. Drops any leading chapter-number prefix like
    "第01章" / "第1章", since the template already shows "CHAPTER N"
    separately above the title — keeping it would just duplicate that.
    Titles are short, so this goes straight to the LLM rather than through
    the local MT + refine pipeline used for chapter bodies.
    """
    if not raw_title or not raw_title.strip():
        return raw_title

    if not contains_chinese(raw_title):
        return raw_title  # already English (or no title at all)

    system_prompt = (
        "Translate this Chinese web novel chapter title into a short, "
        "natural-sounding English chapter title. If the source starts with "
        "a chapter-number marker such as '第01章' or '第1章', drop that "
        "marker entirely and translate only the title text that follows. "
        "Return ONLY the translated title — no quotes, no trailing "
        "punctuation, no chapter number, no explanation."
    )
    translated = call_lm_studio(system_prompt, raw_title, max_tokens=60)
    return translated.strip().strip('"').strip("'").strip()


# ---------------------------------------------------------------------------
# HTML OUTPUT FOR A SINGLE CHAPTER
# ---------------------------------------------------------------------------
CHAPTER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chapter {num} — {title}</title>
<style>
  :root {{
    --bg: #fbf7ee; --text: #2a2620; --accent: #a8763e; --muted: #8a8272;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    max-width: 700px;
    margin: 0 auto;
    padding: 32px 20px 90px;
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 19px;
    line-height: 1.75;
    background: var(--bg);
    color: var(--text);
  }}
  .topline {{
    font-family: -apple-system, sans-serif;
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}
  h1 {{ font-size: 26px; margin: 0 0 30px; color: var(--accent); }}
  p {{ margin: 0 0 1.15em; }}
  .nav {{
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 48px; padding-top: 20px; border-top: 1px solid #e4dcc9;
    font-family: -apple-system, sans-serif; font-size: 15px;
  }}
  .nav a {{ text-decoration: none; color: var(--accent); font-weight: 600; }}
  .nav a.disabled {{ color: #cfc6b2; pointer-events: none; }}
  .home {{ text-align: center; }}
  .home a {{ color: var(--muted); text-decoration: none; font-size: 14px; }}
</style>
</head>
<body>
  <div class="topline">Chapter {num}</div>
  <h1>{title}</h1>
{body}
  <div class="nav">
    <a href="chapter_{prev:03d}.html" class="{prev_class}">&larr; Previous</a>
    <span class="home"><a href="index.html">Table of Contents</a></span>
    <a href="chapter_{next:03d}.html">Next &rarr;</a>
  </div>
</body>
</html>
"""


def write_chapter_html(chapter_num, translated_text, title=None, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    title = title or f"Chapter {chapter_num}"
    paragraphs = [p.strip() for p in translated_text.split("\n") if p.strip()]
    body = "\n".join(f"    <p>{p}</p>" for p in paragraphs)
    prev_num = chapter_num - 1
    html = CHAPTER_TEMPLATE.format(
        num=chapter_num,
        title=title,
        body=body,
        prev=max(prev_num, 1),
        prev_class="disabled" if prev_num < 1 else "",
        next=chapter_num + 1,
    )
    out_path = os.path.join(out_dir, f"chapter_{chapter_num:03d}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


# ---------------------------------------------------------------------------
# AUTO-UPDATING INDEX.HTML (table of contents)
# ---------------------------------------------------------------------------
INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{novel_title}</title>
<style>
  :root {{
    --bg: #fbf7ee; --text: #2a2620; --accent: #a8763e; --muted: #8a8272;
    --card: #ffffff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    max-width: 720px;
    margin: 0 auto;
    padding: 40px 20px 90px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
  }}
  h1 {{
    font-family: Georgia, serif;
    font-size: 30px;
    color: var(--accent);
    margin-bottom: 4px;
  }}
  .subtitle {{ color: var(--muted); font-size: 14px; margin-bottom: 30px; }}
  .chapter-list {{ display: flex; flex-direction: column; gap: 10px; }}
  .chapter-card {{
    background: var(--card);
    border: 1px solid #e9e1cc;
    border-radius: 10px;
    padding: 14px 18px;
    text-decoration: none;
    color: var(--text);
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: transform 0.1s ease, box-shadow 0.1s ease;
  }}
  .chapter-card:hover {{
    transform: translateY(-1px);
    box-shadow: 0 3px 10px rgba(0,0,0,0.06);
    border-color: var(--accent);
  }}
  .chapter-num {{
    font-weight: 700; color: var(--accent); min-width: 70px;
  }}
  .chapter-title {{ flex: 1; padding: 0 12px; }}
  .chapter-date {{ font-size: 12px; color: var(--muted); white-space: nowrap; }}
  .empty {{ color: var(--muted); font-style: italic; padding: 30px 0; }}
</style>
</head>
<body>
  <h1>{novel_title}</h1>
  <div class="subtitle">{count} chapter(s) translated · updated {updated}</div>
  <div class="chapter-list">
{chapter_cards}
  </div>
</body>
</html>
"""

CARD_TEMPLATE = (
    '    <a class="chapter-card" href="chapter_{num:03d}.html">'
    '<span class="chapter-num">Ch. {num}</span>'
    '<span class="chapter-title">{title}</span>'
    '<span class="chapter-date">{date}</span></a>'
)


def regenerate_index(conn, novel_title="My Novel Library", out_dir=OUTPUT_DIR):
    """
    Rebuilds index.html from the chapters table. Call this after every
    single chapter is published (CLI or GUI) so the table of contents
    is always current — this is the "auto-update" behavior.
    """
    c = conn.cursor()
    c.execute("SELECT chapter, title, translated_at FROM chapters ORDER BY chapter ASC")
    rows = c.fetchall()

    if not rows:
        cards_html = '    <div class="empty">No chapters published yet.</div>'
    else:
        cards = []
        for num, title, translated_at in rows:
            date_str = translated_at.split("T")[0] if translated_at else ""
            cards.append(CARD_TEMPLATE.format(
                num=num, title=title or f"Chapter {num}", date=date_str
            ))
        cards_html = "\n".join(cards)

    html = INDEX_TEMPLATE.format(
        novel_title=novel_title,
        count=len(rows),
        updated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        chapter_cards=cards_html,
    )
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)
    return index_path


# ---------------------------------------------------------------------------
# FULL PIPELINE FOR ONE CHAPTER (used by both CLI and GUI)
# ---------------------------------------------------------------------------
def process_chapter(raw_text, chapter_num, title=None, novel_title="My Novel Library"):
    """
    Runs the full pipeline for one chapter and returns a dict with the
    translated text, any flags raised, and the output file path.
    This is the single function the GUI calls.
    """
    conn = init_db()

    title = translate_title(title)

    normalized_text = convert_chinese_numbers(raw_text)
    glossary_text = get_glossary_text(conn)
    translated = translate_chapter(normalized_text, glossary_text)

    extracted = extract_facts_and_entities(translated)
    update_glossary(conn, extracted.get("entities", []), chapter_num)
    flags = check_and_store_facts(conn, extracted.get("facts", []), chapter_num)
    log_flags(flags)

    out_path = write_chapter_html(chapter_num, translated, title=title)

    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO chapters (chapter, title, translated_at) "
        "VALUES (?, ?, ?)",
        (chapter_num, title or f"Chapter {chapter_num}",
         datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()

    index_path = regenerate_index(conn, novel_title=novel_title)
    conn.close()

    return {
        "translated_text": translated,
        "flags": flags,
        "chapter_path": out_path,
        "index_path": index_path,
    }
