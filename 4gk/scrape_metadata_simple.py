import asyncio
import json
import re
import os
import time
import logging
from typing import Optional, Any, Dict
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from fire import Fire

BASE = "https://gotquestions.online"

# ---------- regexes & helpers ----------
RE_INT = re.compile(r"\b(\d{1,6})\b")

# Difficulty patterns, e.g.:
#   "Сложность trueDL 7.7 · 6.2"
#   "Difficulty trueDL 7,7 • 6,2"
RE_DIFFICULTY_LINE = re.compile(
    r"(?:Сложность|Difficulty)\s*(?:trueDL)?\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:[·•/|\\-]\s*([0-9]+(?:[.,][0-9]+)?))?",
    re.I
)

MONTHS_RU = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}

def to_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    try:
        m = RE_INT.search(s.replace("\xa0", " "))
        return int(m.group(1)) if m else None
    except Exception:
        return None

def to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = s.strip().replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def normalize_ru_date(s: Optional[str]) -> Optional[str]:
    """
    Parse Russian dates like '25 сентября 2025 г.' -> '2025-09-25'.
    Also tolerates extra words and punctuation after the year.
    """
    if not s:
        return None
    s = s.strip().lower().replace("года", "").replace("г.", "").replace("год", "")
    m = re.search(r"\b(\d{1,2})\s+([а-яёa-z]+)\s+(\d{4})\b", s, flags=re.I)
    if not m:
        return None
    day = int(m.group(1))
    mon_word = m.group(2).strip(" .").lower()
    year = int(m.group(3))
    month = MONTHS_RU.get(mon_word)
    if not month:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"

def extract_difficulty(body_text: str) -> Dict[str, Any]:
    """
    Extract up to two difficulty numbers from a 'Сложность …' line.
    Returns floats (if parsed) plus the raw snippet for traceability.
    """
    body = body_text.replace("\xa0", " ")
    m = RE_DIFFICULTY_LINE.search(body)
    if not m:
        return {"difficulty_trueDL": None, "difficulty_alt": None, "difficulty_raw": None}
    d1 = to_float(m.group(1))
    d2 = to_float(m.group(2)) if m.group(2) else None
    span = 140
    start = max(0, m.start() - span)
    end = min(len(body), m.end() + span)
    raw = body[start:end].strip()
    return {"difficulty_trueDL": d1, "difficulty_alt": d2, "difficulty_raw": raw}

def extract_pack_meta_from_text(body_text: str) -> Dict[str, Any]:
    """
    From the page visible text, extract:
      - Вопросов N
      - Начало <date>
      - Окончание <date>
      - Опубликован <date>
      - Сложность (via extract_difficulty)
    """
    body = body_text.replace("\xa0", " ")

    def find_after(label_ru: str) -> Optional[str]:
        m = re.search(label_ru + r"\s*([^\n\r]+)", body, flags=re.I)
        return m.group(1).strip() if m else None

    num_questions = to_int(find_after(r"Вопросов"))
    start_raw = find_after(r"Начало")
    end_raw = find_after(r"Окончание")
    publ_raw = find_after(r"Опубликован")

    meta = {
        "num_questions_declared": num_questions,
        "start_date": normalize_ru_date(start_raw),
        "end_date": normalize_ru_date(end_raw),
        "published_date": normalize_ru_date(publ_raw),
    }
    meta.update(extract_difficulty(body))
    return meta

async def auto_scroll(page, max_rounds: int = 6, pause_ms: int = 400):
    """
    Light lazy-load scroller to trigger client-side rendering.
    (Reduced rounds for speed; we only need header text.)
    """
    last_height = 0
    for _ in range(max_rounds):
        await page.evaluate("""() => { window.scrollBy(0, window.innerHeight * 1.2); }""")
        await page.wait_for_timeout(pause_ms)
        height = await page.evaluate("() => document.body.scrollHeight")
        if height == last_height:
            break
        last_height = height

async def read_body_text(page) -> str:
    try:
        return (await page.inner_text("body")).replace("\xa0", " ")
    except Exception:
        return ""

async def extract_pack_title(page) -> str:
    return await page.evaluate(
        """() => {
            const og = document.querySelector('meta[property="og:title"], meta[name="og:title"]');
            if (og && og.content) return og.content.trim();
            const h = document.querySelector('main h1, h1, main h2, h2');
            return h ? h.textContent.trim() : '';
        }"""
    )

# ---------- main parser ----------
async def parse_pack_page(page, pack_id: int) -> Dict[str, Any]:
    pack_url = f"{BASE}/pack/{pack_id}"
    await page.goto(pack_url, wait_until="networkidle")
    await page.wait_for_selector("body", timeout=15000)

    # Allow any client-side rendering or lazy content to appear
    await auto_scroll(page, max_rounds=4, pause_ms=300)

    title = await extract_pack_title(page)
    body_text = await read_body_text(page)
    meta_text = extract_pack_meta_from_text(body_text)

    return {
        "pack_id": pack_id,
        "pack_url": pack_url,
        "pack_title": title or "",
        "num_questions": meta_text.get("num_questions_declared"),
        "start_date": meta_text.get("start_date"),
        "end_date": meta_text.get("end_date"),
        "published_date": meta_text.get("published_date"),
        "difficulty_trueDL": meta_text.get("difficulty_trueDL"),
        "difficulty_alt": meta_text.get("difficulty_alt"),
        "difficulty_raw": meta_text.get("difficulty_raw"),
    }

# ---------- CLI & driver ----------
async def amain(
    pack_wait: float = 0.25,
    start_id: int = 1,     # default to the example pack
    end_id: Optional[int] = None,
    num_packs: int = 1,
    out_dir: str = "pack_meta",
    skip_existing: bool = True,
    verbose: bool = False,
):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    os.makedirs(out_dir, exist_ok=True)
    if not end_id:
        end_id = start_id + num_packs
    pack_ids = list(range(start_id, end_id))[::-1]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            locale="ru-RU",
            viewport={"width": 1366, "height": 900},
        )
        await context.set_extra_http_headers(
            {
                "User-Agent": "gotq-pack-meta-scraper/1.0 (+contact@example.com)",
                "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
            }
        )
        page = await context.new_page()

        for i, pack_id in enumerate(pack_ids, 1):
            out_path = os.path.join(out_dir, f"meta_{pack_id}.json")
            if skip_existing and os.path.exists(out_path):
                logging.info("Pack %s: skip (exists)", pack_id)
                continue

            started = time.perf_counter()
            try:
                data = await parse_pack_page(page, pack_id)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                logging.info("Pack %s: saved %s", pack_id, out_path)
            except PlaywrightTimeoutError as e:
                logging.warning("Pack %s: timeout: %s", pack_id, e)
            except Exception as e:
                logging.exception("Pack %s: failed: %s", pack_id, e)

            elapsed = time.perf_counter() - started
            logging.info("Pack %s: elapsed %.2fs; waiting %.2fs", pack_id, elapsed, pack_wait)
            await asyncio.sleep(pack_wait)

        await browser.close()

def main():
    level = logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    Fire(amain)

if __name__ == "__main__":
    main()
