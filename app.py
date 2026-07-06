#!/usr/bin/env python3
"""
Local GUI for the novel translator.

Run:
    pip install flask requests
    python app.py

Then open http://localhost:8000 in your browser.

- Paste raw Chinese chapter text, set the chapter number/title, click
  Translate. The translated text appears in a box you can copy, AND it
  is automatically saved into output/chapter_XXX.html, with
  output/index.html (the table of contents) regenerated automatically.
- A sidebar lists every chapter published so far, pulled live from the
  database, with a link to open each one.
"""

import os
import sqlite3
from flask import Flask, request, jsonify, render_template_string, send_from_directory

from pipeline import process_chapter, init_db, OUTPUT_DIR

app = Flask(__name__)

NOVEL_TITLE = "My Novel Library"   # change this to your novel's actual title


# ---------------------------------------------------------------------------
# Serve the published library (chapters + index.html) as static files
# ---------------------------------------------------------------------------
@app.route("/library/<path:filename>")
def library_files(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# ---------------------------------------------------------------------------
# API: list published chapters (for the sidebar)
# ---------------------------------------------------------------------------
@app.route("/api/chapters")
def api_chapters():
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT chapter, title, translated_at FROM chapters ORDER BY chapter DESC")
    rows = c.fetchall()
    conn.close()
    return jsonify([
        {"chapter": r[0], "title": r[1], "translated_at": r[2]} for r in rows
    ])


# ---------------------------------------------------------------------------
# API: translate a chapter
# ---------------------------------------------------------------------------
@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json(force=True)
    raw_text = (data.get("raw_text") or "").strip()
    chapter_num = data.get("chapter_num")
    title = (data.get("title") or "").strip() or None

    if not raw_text:
        return jsonify({"error": "Paste the raw Chinese chapter text first."}), 400
    if not chapter_num:
        return jsonify({"error": "Enter a chapter number."}), 400

    try:
        chapter_num = int(chapter_num)
    except ValueError:
        return jsonify({"error": "Chapter number must be a whole number."}), 400

    try:
        result = process_chapter(raw_text, chapter_num, title=title, novel_title=NOVEL_TITLE)
    except Exception as e:
        return jsonify({"error": f"Translation failed: {e}"}), 500

    chapter_filename = os.path.basename(result["chapter_path"])
    return jsonify({
        "translated_text": result["translated_text"],
        "flags": result["flags"],
        "chapter_url": f"/library/{chapter_filename}",
        "index_url": "/library/index.html",
    })


# ---------------------------------------------------------------------------
# Main dashboard page
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Translator Desk — {{ novel_title }}</title>
<style>
  :root {
    --paper: #f6f3ec;
    --ink: #201d18;
    --ink-soft: #6b6459;
    --seal: #3d5a80;
    --seal-dark: #2c4562;
    --warn: #9a3b3b;
    --line: #e2dccb;
    --card: #ffffff;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
  .app {
    display: grid;
    grid-template-columns: 280px 1fr;
    min-height: 100vh;
  }
  /* Sidebar */
  .sidebar {
    background: var(--ink);
    color: var(--paper);
    padding: 28px 20px;
    display: flex;
    flex-direction: column;
  }
  .brand {
    font-family: Georgia, serif;
    font-size: 20px;
    letter-spacing: 0.02em;
    margin-bottom: 2px;
  }
  .brand-sub {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #a39a86;
    margin-bottom: 28px;
  }
  .lib-link {
    display: inline-block;
    font-size: 13px;
    color: var(--paper);
    text-decoration: none;
    border: 1px solid #4a453a;
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 24px;
    text-align: center;
  }
  .lib-link:hover { border-color: #7d7362; }
  .chapter-count {
    font-size: 12px;
    color: #a39a86;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
  }
  .chapter-nav {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .chapter-nav a {
    color: #cfc8b8;
    text-decoration: none;
    font-size: 13px;
    padding: 6px 8px;
    border-radius: 6px;
  }
  .chapter-nav a:hover { background: #322e26; color: var(--paper); }
  .empty-nav { color: #7d7362; font-size: 13px; font-style: italic; }

  /* Main panel */
  .main {
    padding: 40px 48px 80px;
    max-width: 980px;
  }
  h1 {
    font-family: Georgia, serif;
    font-size: 28px;
    margin: 0 0 6px;
  }
  .tagline { color: var(--ink-soft); font-size: 14px; margin-bottom: 32px; }

  .fields-row { display: flex; gap: 16px; margin-bottom: 16px; }
  .field { display: flex; flex-direction: column; gap: 6px; }
  .field label {
    font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--ink-soft);
  }
  .field input {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 14px;
    font-family: inherit;
    background: var(--card);
  }
  .field.chapnum input { width: 100px; }
  .field.title { flex: 1; }

  .panes {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  .pane {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .pane-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 16px;
    border-bottom: 1px solid var(--line);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-soft);
  }
  textarea {
    flex: 1;
    min-height: 420px;
    border: none;
    resize: vertical;
    padding: 16px;
    font-family: Georgia, serif;
    font-size: 15px;
    line-height: 1.6;
    outline: none;
  }
  .out-text { background: #fdfcf9; color: var(--ink); }
  .copy-btn, .translate-btn, .open-btn {
    font-family: inherit;
    font-size: 13px;
    border: none;
    border-radius: 7px;
    padding: 6px 12px;
    cursor: pointer;
  }
  .copy-btn { background: var(--line); color: var(--ink); }
  .copy-btn:hover { background: #d5cdb5; }
  .open-btn { background: var(--line); color: var(--ink); text-decoration: none; display: inline-block; }

  .actions {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-top: 20px;
  }
  .translate-btn {
    background: var(--seal);
    color: white;
    padding: 12px 26px;
    font-size: 15px;
    font-weight: 600;
    border-radius: 9px;
  }
  .translate-btn:hover { background: var(--seal-dark); }
  .translate-btn:disabled { background: #b7b7b7; cursor: not-allowed; }

  .status { font-size: 13px; color: var(--ink-soft); }
  .status.error { color: var(--warn); }

  .flags {
    margin-top: 18px;
    background: #fbeeee;
    border: 1px solid #e6c9c9;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    color: var(--warn);
  }
  .flags strong { display: block; margin-bottom: 6px; }
  .flags ul { margin: 0; padding-left: 18px; }

  @media (max-width: 800px) {
    .app { grid-template-columns: 1fr; }
    .sidebar { display: none; }
    .panes { grid-template-columns: 1fr; }
    .main { padding: 24px; }
  }
</style>
</head>
<body>
<div class="app">
  <div class="sidebar">
    <div class="brand">{{ novel_title }}</div>
    <div class="brand-sub">Translator Desk</div>
    <a class="lib-link" href="/library/index.html" target="_blank">Open reading library &rarr;</a>
    <div class="chapter-count" id="chapterCount">0 chapters published</div>
    <div class="chapter-nav" id="chapterNav">
      <div class="empty-nav">Nothing published yet</div>
    </div>
  </div>

  <div class="main">
    <h1>Paste, translate, publish</h1>
    <div class="tagline">Raw Chinese in, formatted chapter out — saved straight into your library.</div>

    <div class="fields-row">
      <div class="field chapnum">
        <label for="chapterNum">Chapter #</label>
        <input type="number" id="chapterNum" min="1" placeholder="1">
      </div>
      <div class="field title">
        <label for="chapterTitle">Title (optional)</label>
        <input type="text" id="chapterTitle" placeholder="e.g. The Awakening">
      </div>
    </div>

    <div class="panes">
      <div class="pane">
        <div class="pane-header"><span>Raw Chinese text</span></div>
        <textarea id="rawText" placeholder="Paste the raw chapter text here..."></textarea>
      </div>
      <div class="pane">
        <div class="pane-header">
          <span>Translated result</span>
          <button class="copy-btn" onclick="copyOutput()">Copy</button>
        </div>
        <textarea id="outText" class="out-text" placeholder="Translation will appear here..." readonly></textarea>
      </div>
    </div>

    <div class="actions">
      <button class="translate-btn" id="translateBtn" onclick="translateChapter()">Translate &amp; Publish</button>
      <span class="status" id="status"></span>
      <a class="open-btn" id="openChapterLink" href="#" target="_blank" style="display:none;">Open published page &rarr;</a>
    </div>

    <div class="flags" id="flagsBox" style="display:none;"></div>
  </div>
</div>

<script>
async function loadChapters() {
  const res = await fetch('/api/chapters');
  const chapters = await res.json();
  const nav = document.getElementById('chapterNav');
  const count = document.getElementById('chapterCount');
  count.textContent = chapters.length + ' chapter' + (chapters.length === 1 ? '' : 's') + ' published';
  if (chapters.length === 0) {
    nav.innerHTML = '<div class="empty-nav">Nothing published yet</div>';
    return;
  }
  nav.innerHTML = chapters.map(c =>
    `<a href="/library/chapter_${String(c.chapter).padStart(3,'0')}.html" target="_blank">Ch. ${c.chapter} — ${c.title || ''}</a>`
  ).join('');
}

function copyOutput() {
  const out = document.getElementById('outText');
  out.select();
  document.execCommand('copy');
  const status = document.getElementById('status');
  status.textContent = 'Copied to clipboard.';
  status.className = 'status';
}

async function translateChapter() {
  const rawText = document.getElementById('rawText').value;
  const chapterNum = document.getElementById('chapterNum').value;
  const title = document.getElementById('chapterTitle').value;
  const btn = document.getElementById('translateBtn');
  const status = document.getElementById('status');
  const flagsBox = document.getElementById('flagsBox');
  const openLink = document.getElementById('openChapterLink');

  flagsBox.style.display = 'none';
  openLink.style.display = 'none';

  if (!rawText.trim()) {
    status.textContent = 'Paste the raw Chinese text first.';
    status.className = 'status error';
    return;
  }
  if (!chapterNum) {
    status.textContent = 'Enter a chapter number.';
    status.className = 'status error';
    return;
  }

  btn.disabled = true;
  status.textContent = 'Translating locally via LM Studio — this can take a minute...';
  status.className = 'status';

  try {
    const res = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ raw_text: rawText, chapter_num: chapterNum, title: title })
    });
    const data = await res.json();

    if (!res.ok) {
      status.textContent = data.error || 'Something went wrong.';
      status.className = 'status error';
      return;
    }

    document.getElementById('outText').value = data.translated_text;
    status.textContent = 'Published to library.';
    status.className = 'status';
    openLink.href = data.chapter_url;
    openLink.style.display = 'inline-block';

    if (data.flags && data.flags.length > 0) {
      flagsBox.innerHTML = '<strong>⚠ Continuity flags — please review:</strong><ul>' +
        data.flags.map(f => `<li>${f}</li>`).join('') + '</ul>';
      flagsBox.style.display = 'block';
    }

    loadChapters();
  } catch (err) {
    status.textContent = 'Request failed: ' + err;
    status.className = 'status error';
  } finally {
    btn.disabled = false;
  }
}

loadChapters();
</script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML, novel_title=NOVEL_TITLE)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_db()
    print("Translator Desk running at http://localhost:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)
