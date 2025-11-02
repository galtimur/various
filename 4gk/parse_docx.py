import re, json, zipfile
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence, Set
from tqdm import tqdm

import unicodedata

def strip_cyrillic_stress(text: str) -> str:
    """
    Remove Cyrillic stress marks (combining acute/grave and their spacing variants)
    without touching letters like 'й' or 'ё'.

    Examples:
        "[Сти́вен Ро́берт] Гу́ттенберг." -> "[Стивен Роберт] Гуттенберг."
    """
    # Spacing accents sometimes used as stress marks
    spacing_stress = {"\u00B4",  # ´  acute accent
                      "\u02CA"}  # ˊ  modifier letter acute accent
    text = "".join(ch for ch in text if ch not in spacing_stress)

    # Remove only the stress-related combining marks
    combining_stress = {"\u0301",  # ◌́ combining acute accent
                        "\u0341",  # ◌́ combining acute tone mark (deprecated)
                        "\u0300",  # ◌̀ combining grave accent (rarely used for stress)
                        "\u0340"}  # ◌̀ combining grave tone mark (deprecated)

    # Decompose to split base letters from combining marks
    nfd = unicodedata.normalize("NFD", text)
    cleaned = "".join(ch for ch in nfd if ch not in combining_stress)

    # Recompose so characters like Й/й, Ё/ё remain correct
    return unicodedata.normalize("NFC", cleaned)

def clean_stress_in_values(obj):
    """
    Recursively traverse a nested structure and apply strip_cyrillic_stress
    to every *value* that is a str. Dict keys are not modified.

    Supports: dict, list, tuple, set (and subclasses).
    """
    if isinstance(obj, str):
        return strip_cyrillic_stress(obj)

    # Mapping (e.g., dict)
    if isinstance(obj, Mapping):
        return obj.__class__((k, clean_stress_in_values(v)) for k, v in obj.items())

    # Sequence but not str/bytes (e.g., list/tuple)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        # Preserve original type
        t = obj.__class__
        try:
            return t(clean_stress_in_values(x) for x in obj)
        except TypeError:
            # Some sequences (like range) can't be rebuilt from a generator
            return t([clean_stress_in_values(x) for x in obj])

    # Set-like
    if isinstance(obj, Set) and not isinstance(obj, (str, bytes, bytearray)):
        return obj.__class__(clean_stress_in_values(x) for x in obj)

    # Anything else: leave as-is
    return obj


def _read_docx_lines_python_docx(path: str) -> list[str] | None:
    try:
        from docx import Document  # type: ignore
    except Exception:
        return None
    try:
        doc = Document(path)
        lines = []
        for p in doc.paragraphs:
            t = (p.text or "").replace("\xa0", " ").strip()
            lines.append("" if t == "" else t)
        while lines and lines[-1] == "":
            lines.pop()
        return lines
    except Exception:
        return None

def _read_docx_lines_zip_xml(path: str) -> list[str]:
    import xml.etree.ElementTree as ET
    NS = {"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    lines = []
    for p in root.findall(".//w:body/w:p", NS):
        texts = [t.text or "" for t in p.findall(".//w:t", NS)]
        line = "".join(texts).replace("\xa0"," ").strip()
        lines.append(line)
    while lines and lines[-1] == "":
        lines.pop()
    return lines

KEY_PATTERNS = {
    "answer": r"^(Ответ)\s*:?\s*(.*)$",
    "grading": r"^(Зач[её]т)\s*:?\s*(.*)$",
    "comment": r"^(Комментарий)\s*:?\s*(.*)$",
    "sources": r"^(Источники?|Источник\(и\)?)\s*:?\s*(.*)$",
    "author": r"^(Автор)\s*:?\s*(.*)$",
}

HEADER_QUESTION_RE = re.compile(r"^Вопрос\s+(\d+)\b", re.IGNORECASE)
HEADER_TOUR_RE = re.compile(r"^Тур\s+(\d+)\b", re.IGNORECASE)
META_EDITORS_RE = re.compile(r"^Редакторы?:\s*(.+)$", re.IGNORECASE)

def _match_key(line: str):
    for key, pat in KEY_PATTERNS.items():
        m = re.match(pat, line, flags=re.IGNORECASE)
        if m:
            return key, m.group(2).strip()
    return None, None

def parse_questions_from_lines(lines: list[str]) -> dict[str, Any]:
    pack: dict[str, Any] = {"title": None, "editors": [], "tours": [], "raw_preamble": []}
    questions: list[dict[str, Any]] = []

    current_tour = None
    title_set = False
    current_q: dict[str, Any] | None = None
    current_field: str | None = None

    def finalize_question():
        nonlocal current_q, questions
        if not current_q:
            return
        for f in ["question","answer","grading","comment","author"]:
            if f in current_q and isinstance(current_q[f], list):
                current_q[f] = "\n".join([x for x in current_q[f] if x.strip()]).strip()
        if "sources" in current_q:
            raw = "\n".join(current_q["sources"]) if isinstance(current_q["sources"], list) else current_q["sources"]
            srcs = [s.strip() for s in re.split(r"\n+", raw) if s.strip()]
            current_q["sources"] = srcs
        questions.append(current_q)
        current_q = None

    for raw in lines:
        line = raw.strip()

        if not title_set and line:
            pack["title"] = line
            title_set = True
            continue

        m_ed = META_EDITORS_RE.match(line)
        if m_ed and not questions:
            editors = [e.strip(" .") for e in re.split(r",|;", m_ed.group(1)) if e.strip()]
            pack["editors"] = editors
            pack["raw_preamble"].append(raw)
            continue

        m_tour = HEADER_TOUR_RE.match(line)
        if m_tour:
            current_tour = int(m_tour.group(1))
            if current_tour not in pack["tours"]:
                pack["tours"].append(current_tour)
            pack["raw_preamble"].append(raw)
            current_field = None
            continue

        m_q = HEADER_QUESTION_RE.match(line)
        if m_q:
            finalize_question()
            current_q = {"index": int(m_q.group(1)), "tour": current_tour, "question": []}
            current_field = "question"
            continue

        if current_q is None:
            pack["raw_preamble"].append(raw)
            continue

        key, remainder = _match_key(line)
        if key:
            current_field = key
            if key == "sources":
                current_q.setdefault("sources", [])
                if remainder:
                    current_q["sources"].append(remainder)
            else:
                current_q[key] = []
                if remainder:
                    current_q[key].append(remainder)
            continue

        if line == "":
            if current_field:
                current_q.setdefault(current_field, []).append("")
            continue

        if current_field:
            current_q.setdefault(current_field, []).append(line)
        else:
            current_q.setdefault("question", []).append(line)

    finalize_question()
    if isinstance(pack["raw_preamble"], list):
        pack["raw_preamble"] = "\n".join([x for x in pack["raw_preamble"] if x is not None])
    pack["questions"] = questions
    return pack

def parse_docx_to_json(path: str, pack_id: int, out_path: str | None = None) -> dict[str, Any]:
    lines = _read_docx_lines_python_docx(path) or _read_docx_lines_zip_xml(path)
    # normalize blanks
    norm = []
    prev_blank = False
    for l in lines:
        s = l.replace("\xa0"," ").strip()
        if s == "":
            if not prev_blank:
                norm.append("")
            prev_blank = True
        else:
            norm.append(s); prev_blank = False
    data = parse_questions_from_lines(norm)
    data["pack_id"] = pack_id

    data = clean_stress_in_values(data)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return data

def process_docx_files(indir: str, outdir: str=  "."):
    inpath = Path(indir)
    outpath = Path(outdir)
    output_file = outpath / "all_data.jsonl"

    docx_files = list(inpath.glob("*.docx"))
    for docx_path in tqdm(docx_files, desc="Processing DOCX files", total=len(docx_files)):
        pack_id = int(docx_path.stem)  # gets filename without extension
        data = parse_docx_to_json(str(docx_path), pack_id)
        questions = []
        for q in data["questions"]:
            if not (q.get("question") and q.get("answer")):
                continue
            answer = q.get("answer")
            q["answer"] = answer.replace('"', '')
            grading = q.get("grading")
            if grading:
                q["grading"] = grading.replace('"', '')
            questions.append(q)
        if not questions:
            continue
        data["num_questions"] = len(questions)
        data["questions"] = questions
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
