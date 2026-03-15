"""
Converts bible_text_kjv.txt (Project Gutenberg KJV) into a clean file where
every line is one verse:  Book Chapter:Verse text

Handles:
  - Gutenberg header/footer stripping
  - Long-form book names → short names  (e.g. "The First Book of Moses: Called Genesis" → Genesis)
  - Line-wrapped verses joined into a single line
  - Inline verse tags  (e.g. "...years: 1:15 And let them..." → split into two verses)
"""

import re

BOOK_MAP = {
    "The First Book of Moses: Called Genesis": "Genesis",
    "The Second Book of Moses: Called Exodus": "Exodus",
    "The Third Book of Moses: Called Leviticus": "Leviticus",
    "The Fourth Book of Moses: Called Numbers": "Numbers",
    "The Fifth Book of Moses: Called Deuteronomy": "Deuteronomy",
    "The Book of Joshua": "Joshua",
    "The Book of Judges": "Judges",
    "The Book of Ruth": "Ruth",
    "The First Book of Samuel": "1 Samuel",
    "The Second Book of Samuel": "2 Samuel",
    "The First Book of the Kings": "1 Kings",
    "The Second Book of the Kings": "2 Kings",
    "The First Book of the Chronicles": "1 Chronicles",
    "The Second Book of the Chronicles": "2 Chronicles",
    "Ezra": "Ezra",
    "The Book of Nehemiah": "Nehemiah",
    "The Book of Esther": "Esther",
    "The Book of Job": "Job",
    "The Book of Psalms": "Psalms",
    "The Proverbs": "Proverbs",
    "Ecclesiastes": "Ecclesiastes",
    "The Song of Solomon": "Song of Solomon",
    "The Book of the Prophet Isaiah": "Isaiah",
    "The Book of the Prophet Jeremiah": "Jeremiah",
    "The Lamentations of Jeremiah": "Lamentations",
    "The Book of the Prophet Ezekiel": "Ezekiel",
    "The Book of Daniel": "Daniel",
    "Hosea": "Hosea",
    "Joel": "Joel",
    "Amos": "Amos",
    "Obadiah": "Obadiah",
    "Jonah": "Jonah",
    "Micah": "Micah",
    "Nahum": "Nahum",
    "Habakkuk": "Habakkuk",
    "Zephaniah": "Zephaniah",
    "Haggai": "Haggai",
    "Zechariah": "Zechariah",
    "Malachi": "Malachi",
    "The Gospel According to Saint Matthew": "Matthew",
    "The Gospel According to Saint Mark": "Mark",
    "The Gospel According to Saint Luke": "Luke",
    "The Gospel According to Saint John": "John",
    "The Acts of the Apostles": "Acts",
    "The Epistle of Paul the Apostle to the Romans": "Romans",
    "The First Epistle of Paul the Apostle to the Corinthians": "1 Corinthians",
    "The Second Epistle of Paul the Apostle to the Corinthians": "2 Corinthians",
    "The Epistle of Paul the Apostle to the Galatians": "Galatians",
    "The Epistle of Paul the Apostle to the Ephesians": "Ephesians",
    "The Epistle of Paul the Apostle to the Philippians": "Philippians",
    "The Epistle of Paul the Apostle to the Colossians": "Colossians",
    "The First Epistle of Paul the Apostle to the Thessalonians": "1 Thessalonians",
    "The Second Epistle of Paul the Apostle to the Thessalonians": "2 Thessalonians",
    "The First Epistle of Paul the Apostle to Timothy": "1 Timothy",
    "The Second Epistle of Paul the Apostle to Timothy": "2 Timothy",
    "The Epistle of Paul the Apostle to Titus": "Titus",
    "The Epistle of Paul the Apostle to Philemon": "Philemon",
    "The Epistle of Paul the Apostle to the Hebrews": "Hebrews",
    "The General Epistle of James": "James",
    "The First Epistle General of Peter": "1 Peter",
    "The Second General Epistle of Peter": "2 Peter",
    "The First Epistle General of John": "1 John",
    "The Second Epistle General of John": "2 John",
    "The Third Epistle General of John": "3 John",
    "The General Epistle of Jude": "Jude",
    "The Revelation of Saint John the Divine": "Revelation",
    # Section headers to ignore
    "The Old Testament of the King James Version of the Bible": None,
    "The New Testament of the King James Bible": None,
}

# Matches an inline verse tag like  1:5  or  12:34
VERSE_TAG = re.compile(r'\b(\d+:\d+)\s+')

GUTENBERG_END = "*** END OF THE PROJECT GUTENBERG"


def split_on_inline_verses(text: str) -> list[tuple[str, str]]:
    """
    Split a text that may contain inline verse tags.
    Returns list of (verse_ref, text_fragment) tuples.
    The first element has ref='' if the line starts with body text.
    """
    parts = []
    remaining = text
    while True:
        m = VERSE_TAG.search(remaining)
        if not m:
            parts.append(("", remaining))
            break
        before = remaining[: m.start()].strip()
        if before:
            parts.append(("", before))
        ref = m.group(1)
        remaining = remaining[m.end():]
        # grab text until next verse tag or end
        m2 = VERSE_TAG.search(remaining)
        if m2:
            parts.append((ref, remaining[: m2.start()].strip()))
            remaining = remaining[m2.start():]
        else:
            parts.append((ref, remaining.strip()))
            break
    return parts


def main():
    input_path = "bible_text_kjv.txt"
    output_path = "bible_kjv_clean.txt"

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    verses = []           # final output lines
    current_book = ""
    current_ref = ""      # e.g. "3:16"
    current_text = ""     # accumulated text for current verse
    in_content = False    # True once we pass the ToC and reach real Bible text

    def flush():
        nonlocal current_ref, current_text
        if current_book and current_ref and current_text.strip():
            chapter, verse = current_ref.split(":")
            verses.append(f"{current_book} {chapter}:{verse} {current_text.strip()}")
        current_ref = ""
        current_text = ""

    for raw in raw_lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        # Stop at Gutenberg footer
        if stripped.startswith(GUTENBERG_END):
            break

        # Skip blank lines
        if not stripped:
            continue

        # Detect book headers (from BOOK_MAP)
        if stripped in BOOK_MAP:
            short = BOOK_MAP[stripped]
            if short is None:
                # Section header (Old/New Testament label) — skip
                continue
            # Real book transition
            flush()
            current_book = short
            in_content = True
            continue

        if not in_content:
            continue

        # Check if line starts with a verse tag like "1:1 In the beginning"
        m = re.match(r'^(\d+:\d+)\s+(.*)', stripped)
        if m:
            flush()
            current_ref = m.group(1)
            rest = m.group(2)
            # Rest may itself contain more inline verse tags
            parts = split_on_inline_verses(rest)
            first = True
            for ref, text in parts:
                if first:
                    current_text = text
                    first = False
                else:
                    if ref:
                        flush()
                        current_ref = ref
                        current_text = text
                    else:
                        current_text += " " + text
        else:
            # Continuation line — may still have inline verse tags
            parts = split_on_inline_verses(stripped)
            first = True
            for ref, text in parts:
                if first and not ref:
                    # Pure continuation
                    current_text += " " + text
                    first = False
                else:
                    if first:
                        first = False
                    flush()
                    current_ref = ref
                    current_text = text

    flush()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(verses) + "\n")

    print(f"Done. {len(verses)} verses written to {output_path}")


if __name__ == "__main__":
    main()
