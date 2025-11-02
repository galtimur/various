import asyncio
import json
import re
import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Any, Dict, List
from urllib.parse import urljoin
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from fire import Fire

BASE = "https://gotquestions.online"

# ---------- regexes & helpers ----------
RE_Q_ID = re.compile(r"/question/(\d+)")
RE_NUM = re.compile(r"\d+")
RE_INT = re.compile(r"\b(\d{1,6})\b")
RE_VOPROS_ORD = re.compile(r"Вопрос\s*#?\s*(\d+)", re.I)

# Difficulty patterns:
#   "Сложность trueDL 7.7 · 6.2"
#   "Difficulty trueDL 7,7 • 6,2"
RE_DIFFICULTY_LINE = re.compile(r"(?:Сложность|Difficulty)\s*(?:trueDL)?\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:[·•/|\\-]\s*([0-9]+(?:[.,][0-9]+)?))?", re.I)

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
    Extracts up to two difficulty numbers from a 'Сложность …' line.
    Returns floats if possible, plus the raw line.
    """
    body = body_text.replace("\xa0", " ")
    m = RE_DIFFICULTY_LINE.search(body)
    if not m:
        return {"difficulty_trueDL": None, "difficulty_alt": None, "difficulty_raw": None}
    d1 = to_float(m.group(1))
    d2 = to_float(m.group(2)) if m.group(2) else None
    # Try to capture the whole nearby 'Сложность …' phrase for traceability
    raw = None
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
        # capture the first non-empty run after the label up to a linebreak
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

async def auto_scroll(page, max_rounds: int = 12, pause_ms: int = 600):
    """
    Lazy-load scroller: scrolls down in steps to trigger client-side rendering.
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

async def extract_pack_counters(page) -> Dict[str, Optional[int]]:
    """
    Attempts to read pack-level like/dislike counters near the title/header area.
    Strategy:
      1) Determine the vertical top of the first question link.
      2) Among all like/dislike icons, keep those located visually ABOVE that top (header area).
      3) In that header subset, pick the first counts we can parse.
    """
    return await page.evaluate(
        """() => {
          function numFrom(el) {
            if (!el) return null;
            // Prefer next sibling
            const ns = el.nextElementSibling;
            let t = '';
            if (ns && /\\d/.test(ns.textContent||'')) t = ns.textContent;
            if (!t) {
              const p = el.closest('button, .v-btn, .counter, .like, .dislike') || el.parentElement;
              if (p) t = p.textContent || '';
            }
            const m = (t||'').replace(/\\u00a0/g, ' ').match(/(\\d{1,6})/);
            return m ? parseInt(m[1], 10) : null;
          }

          const firstQ = document.querySelector('a[href^="/question/"]');
          const qTop = firstQ ? firstQ.getBoundingClientRect().top : Infinity;

          const icons = Array.from(document.querySelectorAll('i, .material-icons, .v-icon'));
          const headerIcons = icons.filter(el => {
            const t = (el.textContent || '').trim().toLowerCase();
            if (!t) return false;
            const top = el.getBoundingClientRect().top;
            return top < qTop - 6 && (t.includes('thumb_up') || t.includes('thumb_down'));
          });

          let like = null, dislike = null;
          for (const el of headerIcons) {
            const t = (el.textContent || '').trim().toLowerCase();
            if (t.includes('thumb_up') && like === null) like = numFrom(el);
            if (t.includes('thumb_down') && dislike === null) dislike = numFrom(el);
            if (like !== null && dislike !== null) break;
          }
          return { pack_likes: like, pack_dislikes: dislike };
        }"""
    )

async def extract_questions_on_pack(page) -> List[Dict[str, Any]]:
    """
    Build the question list from the pack page without opening each question.
    Dedupes by qid. Extracts ordinal, like, dislike.
    """
    questions = await page.evaluate(
        """() => {
            const seen = new Set();
            const out = [];

            function parseCounters(container) {
              function findCounter(iconNames) {
                const icons = Array.from(container.querySelectorAll('i, .material-icons, .v-icon'));
                for (const el of icons) {
                  const txt = (el.textContent || '').trim().toLowerCase();
                  if (!txt) continue;
                  if (iconNames.some(n => txt.includes(n))) {
                    // Prefer a sibling
                    const ns = el.nextElementSibling;
                    let src = '';
                    if (ns && /\\d/.test(ns.textContent||'')) src = ns.textContent;
                    if (!src) {
                      const p = el.closest('button, .v-btn, .counter, .like, .dislike') || el.parentElement;
                      if (p) src = p.textContent || '';
                    }
                    const m = (src||'').replace(/\\u00a0/g, ' ').match(/(\\d{1,6})/);
                    if (m) return parseInt(m[1], 10);
                  }
                }
                // fallback: coarse scan
                const near = (container.innerText || '').replace(/\\u00a0/g, ' ');
                for (const name of iconNames) {
                  const re = new RegExp(name + '\\\\s*(\\\\d{1,6})', 'i');
                  const m = near.match(re);
                  if (m) return parseInt(m[1], 10);
                }
                return null;
              }
              const likes = findCounter(['thumb_up','thumb_up_alt','favorite']);
              const dislikes = findCounter(['thumb_down','thumb_down_alt']);
              return { likes, dislikes };
            }

            // Strategy 1: use anchors to /question/<id>
            const anchors = Array.from(document.querySelectorAll('a[href^="/question/"]'));
            for (const a of anchors) {
              const href = a.getAttribute('href') || '';
              const m = href.match(/\\/question\\/(\\d+)/);
              if (!m) continue;
              const qid = parseInt(m[1], 10);
              if (seen.has(qid)) continue;
              seen.add(qid);

              let container = a.closest('article, .question, .qa-card, .card, .v-list-item, .v-sheet, li, div');
              if (!container) container = a.parentElement || document.body;

              const txt = (container.textContent || '').trim();
              let ordinal = null;
              const om = txt.match(/Вопрос\\s*#?\\s*(\\d+)/i);
              if (om) ordinal = parseInt(om[1], 10);

              const { likes, dislikes } = parseCounters(container);
              out.push({
                qid,
                url: new URL(href, location.origin).href,
                ordinal_in_pack: ordinal,
                likes,
                dislikes
              });
            }

            // Strategy 2 (fallback): blocks with data-question-id etc., if anchors were missing
            if (out.length === 0) {
              const blocks = Array.from(document.querySelectorAll('[data-question-id], [id^="question-"], .question'));
              for (const b of blocks) {
                let qid = null, link = null;
                const a = b.querySelector('a[href^="/question/"]');
                if (a) {
                  const href = a.getAttribute('href') || '';
                  const m = href.match(/\\/question\\/(\\d+)/);
                  if (m) { qid = parseInt(m[1], 10); link = new URL(href, location.origin).href; }
                }
                const txt = (b.textContent || '').trim();
                let ordinal = null;
                const om = txt.match(/Вопрос\\s*#?\\s*(\\d+)/i);
                if (om) ordinal = parseInt(om[1], 10);
                const { likes, dislikes } = parseCounters(b);
                out.push({ qid, url: link, ordinal_in_pack: ordinal, likes, dislikes });
              }
            }

            return out;
        }"""
    )
    # de-duplicate a second time (paranoia) and sort by ordinal if present
    seen = set()
    unique: List[Dict[str, Any]] = []
    for q in questions:
        key = q.get("qid") or q.get("url") or q.get("ordinal_in_pack")
        if key in seen:
            continue
        seen.add(key)
        unique.append(q)

    def ord_key(x):  # stable sort: ordinal, then qid
        o = x.get("ordinal_in_pack")
        qid = x.get("qid") or 10**12
        return (o if isinstance(o, int) else 10**12, qid)
    unique.sort(key=ord_key)
    return unique

# ---------- main parser ----------
async def parse_pack_page(page, pack_id: int) -> Dict[str, Any]:
    pack_url = f"{BASE}/pack/{pack_id}"
    await page.goto(pack_url, wait_until="networkidle")
    await page.wait_for_selector("body", timeout=15000)

    # Allow any client-side rendering or lazy content to appear
    await auto_scroll(page, max_rounds=10, pause_ms=500)

    title = await extract_pack_title(page)
    header_counts = await extract_pack_counters(page)
    body_text = await read_body_text(page)
    meta_text = extract_pack_meta_from_text(body_text)
    questions = await extract_questions_on_pack(page)

    # prefer declared num if present, else count we collected
    num_declared = meta_text.get("num_questions_declared")
    num_detected = len(questions)
    num_questions = num_declared if isinstance(num_declared, int) else num_detected

    return {
        "pack_id": pack_id,
        "pack_url": pack_url,
        "pack_title": title or "",
        "pack_likes": header_counts.get("pack_likes"),
        "pack_dislikes": header_counts.get("pack_dislikes"),
        "num_questions": num_questions,
        "start_date": meta_text.get("start_date"),
        "end_date": meta_text.get("end_date"),
        "published_date": meta_text.get("published_date"),
        "difficulty_trueDL": meta_text.get("difficulty_trueDL"),
        "difficulty_alt": meta_text.get("difficulty_alt"),
        "difficulty_raw": meta_text.get("difficulty_raw"),
        # Question list excludes question text/answer entirely
        "questions": questions,
    }

# ---------- CLI & driver ----------
async def amain(
    pack_wait: float = 1.0,
    start_id: int = 6525,     # default to the example pack
    num_packs: int = 1,
    out_dir: str = ".",
    skip_existing: bool = True,
    verbose: bool = False,
):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    os.makedirs(out_dir, exist_ok=True)
    pack_ids = list(range(start_id, start_id + num_packs))

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
            out_path = os.path.join(out_dir, f"pack_{pack_id}_meta.json")
            if skip_existing and os.path.exists(out_path):
                logging.info("Pack %s: skip (exists)", pack_id)
                continue

            started = time.perf_counter()
            try:
                data = await parse_pack_page(page, pack_id)
                # sanity: do not include question texts/answers
                # (the extractor never collects them)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logging.info(
                    "Pack %s: saved %s (%d questions)",
                    pack_id, out_path, len(data.get("questions", []))
                )
            except PlaywrightTimeoutError as e:
                logging.warning("Pack %s: timeout: %s", pack_id, e)
            except Exception as e:
                logging.exception("Pack %s: failed: %s", pack_id, e)

            elapsed = time.perf_counter() - started
            logging.info(
                "Pack %s: elapsed %.2fs; waiting %.2fs before next pack",
                pack_id, elapsed, pack_wait
            )
            await asyncio.sleep(pack_wait)

        await browser.close()

def main():
    level = logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    Fire(amain)

if __name__ == "__main__":
    main()
