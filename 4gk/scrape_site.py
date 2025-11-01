import asyncio
import json
import re
import os
import time
import logging
from urllib.parse import urljoin, urlparse
from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
)
from fire import Fire

BASE = "https://gotquestions.online"

# ---------- regexes & helpers ----------
RE_TRUEDL = re.compile(
    r"(?:Сложность|Difficulty)\s*(?:trueDL)?\s*([0-9]+(?:[.,][0-9]+)?)", re.I
)
RE_Q_ID = re.compile(r"/question/(\d+)")


def to_int(s: str | None):
    if not s:
        return None
    try:
        return int(re.sub(r"[^\d]", "", s))
    except Exception:
        return None


async def get_like_count(page):
    """
    Heuristic: look for Material Icons 'thumb_up' and read a nearby number.
    Falls back to scanning visible text.
    """
    # Pattern 1: <i>thumb_up</i> followed by a number
    icon = page.locator('i:has-text("thumb_up")')
    if await icon.count():
        num = await icon.first().evaluate(
            """(el) => {
            const next = el.nextElementSibling?.textContent?.trim();
            if (next && /\d/.test(next)) return next;
            const p = el.parentElement;
            const maybe = p && p.querySelector("span,b,strong,.count,.value");
            return maybe?.textContent?.trim() || null;
        }"""
        )
        if num:
            return to_int(num)

    # Pattern 2: aria labels or titles
    labelled = page.locator(
        '[aria-label*="thumb_up"], [title*="thumb_up"], [aria-label*="лайк"], [title*="лайк"]'
    )
    if await labelled.count():
        num = await labelled.first().evaluate(
            """(el) => {
            const s = el.closest('*');
            if (!s) return null;
            const m = (s.textContent||'').match(/thumb_up\\s*(\\d+)/) || (s.textContent||'').match(/\\b(\\d+)\\b/);
            return m ? m[1] : null;
        }"""
        )
        if num:
            return to_int(num)

    # Pattern 3: coarse fallback
    txt = (await page.inner_text("body")).replace("\xa0", " ")
    m = re.search(r"thumb_up\s*([0-9]{1,6})", txt)
    if m:
        return to_int(m.group(1))
    return None


async def find_trueDL(page):
    body = (await page.inner_text("body")).replace("\xa0", " ")
    m = RE_TRUEDL.search(body)
    if m:
        return float(m.group(1).replace(",", "."))
    return None


def extract_ids_from_hrefs(hrefs):
    out, seen = [], set()
    for href in hrefs:
        if not href:
            continue
        m = RE_Q_ID.search(href)
        if not m:
            continue
        qid = int(m.group(1))
        if qid in seen:
            continue
        seen.add(qid)
        out.append((qid, urljoin(BASE, href)))
    return out


# ---------- page parsers ----------
async def parse_pack(page, pack_url):
    await page.goto(pack_url, wait_until="networkidle")
    try:
        # If a pack has no question links, don't fail the whole parse
        await page.wait_for_selector('a[href^="/question/"]', timeout=7000)
    except PlaywrightTimeoutError:
        pass

    # Title from og:title or first H1/H2
    title = await page.evaluate(
        """() => {
        const m = document.querySelector('meta[property="og:title"], meta[name="og:title"]');
        if (m && m.content) return m.content.trim();
        const h = document.querySelector('h1, h2');
        return h ? h.textContent.trim() : '';
    }"""
    )

    pack_likes = await get_like_count(page)
    truedl = await find_trueDL(page)

    # Collect question links; try to scroll a bit for lazy‑load
    hrefs = await page.locator('a[href^="/question/"]').evaluate_all(
        "els => els.map(a => a.getAttribute('href'))"
    )
    questions = extract_ids_from_hrefs(hrefs)

    prev = len(questions)
    for _ in range(6):  # up to 6 lazy-load batches
        await page.mouse.wheel(delta_x=0, delta_y=20000)
        await page.wait_for_timeout(600)
        hrefs = await page.locator('a[href^="/question/"]').evaluate_all(
            "els => els.map(a => a.getAttribute('href'))"
        )
        questions = extract_ids_from_hrefs(hrefs)
        if len(questions) == prev:
            break
        prev = len(questions)

    return {
        "pack_title": title or "",
        "pack_likes": pack_likes,
        "difficulty_trueDL": truedl,
        "question_links": [url for _, url in questions],
    }


async def parse_question(page, q_url):
    await page.goto(q_url, wait_until="networkidle")
    await page.wait_for_selector("body", timeout=15000)

    # ID from URL
    m = RE_Q_ID.search(urlparse(q_url).path)
    qid = int(m.group(1)) if m else None

    title = await page.evaluate(
        """() => {
        const h = document.querySelector('h1, h2, h3');
        return h ? h.textContent.trim() : '';
    }"""
    )

    likes = await get_like_count(page)

    # Grab main text
    text = await page.evaluate(
        """() => {
        const main = document.querySelector('main') ||
                     document.querySelector('[role="main"], .content, .question, article, .wrapper');
        return (main || document.body).innerText.trim();
    }"""
    )

    ord_m = re.search(r"Вопрос\s*#?\s*(\d+)", title or "", flags=re.I)
    ordinal = int(ord_m.group(1)) if ord_m else None

    return {
        "qid": qid,
        "title": title,
        "ordinal_in_pack": ordinal,
        "likes": likes,
        "text": text,
    }


async def scrape_single_pack(page, pack_id: int, question_wait: float):
    pack_url = f"{BASE}/pack/{pack_id}"
    logging.info("Pack %s: opening %s", pack_id, pack_url)

    pack = await parse_pack(page, pack_url)
    results = {
        "pack_id": pack_id,
        "pack_url": pack_url,
        "pack_title": pack["pack_title"],
        "pack_likes": pack["pack_likes"],
        "difficulty_trueDL": pack["difficulty_trueDL"],
        "questions": [],
    }

    for q_url in pack["question_links"]:
        try:
            qdata = await parse_question(page, q_url)
            results["questions"].append({**qdata, "url": q_url})
        except Exception as e:
            logging.warning("Pack %s: question %s failed: %s", pack_id, q_url, e)
        # be nice to the site between questions
        await asyncio.sleep(max(0.0, question_wait))
    return results


# ---------- CLI & driver ----------
async def amain(
    pack_wait: float = 1.0, # wait between packs
    q_wait: float = 0.1, # wait between questions
    start_id: int = 1, # start from this pack id
    num_packs: int = 1, # number of packs to scrape
    out_dir: str = ".", # output directory
    skip_existing: bool = True,
    verbose: bool = False,
):
    # pack_wait = max(4.0, pack_wait)
    # q_wait = max(0.0, q_wait)
    pack_ids = list(range(start_id, start_id + num_packs))
    os.makedirs(out_dir, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            locale="ru-RU", viewport={"width": 1366, "height": 900}
        )
        await context.set_extra_http_headers(
            {
                "User-Agent": "gotq-scraper/1.0 (+contact@example.com)",
                "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
            }
        )
        page = await context.new_page()

        for i, pack_id in enumerate(pack_ids, 1):
            # continue
            print(pack_id)
            out_path = os.path.join(out_dir, f"pack_{pack_id}.json")
            if skip_existing and os.path.exists(out_path):
                logging.info("Pack %s: skip (exists)", pack_id)
                continue

            started = time.perf_counter()
            try:
                data = await scrape_single_pack(page, pack_id, question_wait=q_wait)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logging.info(
                    "Pack %s: saved %s (%d questions)",
                    pack_id,
                    out_path,
                    len(data.get("questions", [])),
                )
            except PlaywrightTimeoutError as e:
                logging.warning("Pack %s: timeout: %s", pack_id, e)
            except Exception as e:
                logging.exception("Pack %s: failed: %s", pack_id, e)

            # Graceful pause between packs (>= 4 seconds)
            elapsed = time.perf_counter() - started
            logging.info(
                "Pack %s: elapsed %.2fs; waiting %.2fs before next pack",
                pack_id,
                elapsed,
                pack_wait,
            )
            await asyncio.sleep(pack_wait)

        await browser.close()


if __name__ == "__main__":
    level = logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )
    Fire(amain)
