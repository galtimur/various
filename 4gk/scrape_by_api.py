import time
import random
import os
import re
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv
from fire import Fire

load_dotenv()

BASE = "https://gotquestions.online"
JWT  = os.environ.get("GQ_JWT")
if not JWT: sys.exit("Set GQ_JWT first.")


def fname_from_cd(cd, fallback):
    if not cd: return fallback
    import urllib.parse
    # handle filename*=UTF-8''name.docx
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd)
    if m: return urllib.parse.unquote(m.group(1))
    m = re.search(r'filename="([^"]+)"', cd)
    if m: return m.group(1)
    return fallback

def main(start_id: int = 5000,
         num_packs: int = 2,
         end_id: int | None = None,
         outdir: str = "downloads"):

    if not end_id:
        end_id = start_id + num_packs
    ids = list(range(start_id, end_id+1))

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    sess = requests.Session()
    sess.headers.update({"Authorization": f"JWT {JWT}", "Accept": "*/*"})

    for pack_id in ids:
        path = outdir / (str(pack_id) + ".docx")
        if path.exists():
            continue
        url = f"{BASE}/api/download/{pack_id}/"
        r = sess.get(url, allow_redirects=True, timeout=60)
        if r.status_code != 200:
            print(f"[{pack_id}] HTTP {r.status_code}: {r.text[:200]}")
            if r.status_code == 401:
                break
            continue
        path.write_bytes(r.content)
        print(f"[{pack_id}] saved -> {path}")
        time.sleep(random.uniform(0.5, 2.0))

if __name__ == "__main__":
    Fire(main)