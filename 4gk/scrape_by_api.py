import time, random
import os, re, sys
import pathlib, requests
from dotenv import load_dotenv

load_dotenv()

BASE = "https://gotquestions.online"
JWT  = os.environ.get("GQ_JWT")
if not JWT: sys.exit("Set GQ_JWT first.")

ids = list(range(5034, 6527))

outdir = pathlib.Path("downloads"); outdir.mkdir(exist_ok=True)
sess = requests.Session()
sess.headers.update({"Authorization": f"JWT {JWT}", "Accept": "*/*"})

def fname_from_cd(cd, fallback):
    if not cd: return fallback
    import urllib.parse
    # handle filename*=UTF-8''name.docx
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd)
    if m: return urllib.parse.unquote(m.group(1))
    m = re.search(r'filename="([^"]+)"', cd)
    if m: return m.group(1)
    return fallback

for pack_id in ids:
    path = outdir / (str(pack_id) + ".docx")
    if path.exists():
        continue
    url = f"{BASE}/api/download/{pack_id}/"
    r = sess.get(url, allow_redirects=True, timeout=60)
    if r.status_code != 200:
        print(f"[{pack_id}] HTTP {r.status_code}: {r.text[:200]}")
        continue
    name = fname_from_cd(r.headers.get("content-disposition"), f"pack_{pack_id}.docx")
    safe = re.sub(r'[\\/:*?"<>|]+', "_", name)
    path.write_bytes(r.content)
    print(f"[{pack_id}] saved -> {path}")
    time.sleep(random.uniform(0.5, 2.0))
