#!/usr/bin/env python3
"""
CLI usage (unchanged from before):
    python translate.py raw_chapters/chapter_001.txt 1
    python translate.py raw_chapters/chapter_002.txt 2 "The Awakening"

Args:
    1) path to raw Chinese .txt file
    2) chapter number
    3) (optional) chapter title
"""

import sys
import os
from pipeline import process_chapter


def main():
    if len(sys.argv) < 3:
        print("Usage: python translate.py <raw_chapter.txt> <chapter_number> [title]")
        sys.exit(1)

    raw_path = sys.argv[1]
    chapter_num = int(sys.argv[2])
    title = sys.argv[3] if len(sys.argv) > 3 else None

    if not os.path.exists(raw_path):
        print(f"File not found: {raw_path}")
        sys.exit(1)

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"[{chapter_num}] Processing...")
    result = process_chapter(raw_text, chapter_num, title=title)

    print(f"\n✅ Done. Chapter file: {result['chapter_path']}")
    print(f"   Index updated: {result['index_path']}")
    if result["flags"]:
        print(f"⚠ {len(result['flags'])} flag(s) logged to flags.log — please review:")
        for flag in result["flags"]:
            print(f"   - {flag}")


if __name__ == "__main__":
    main()
