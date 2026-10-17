#!/usr/bin/env python3
"""Download the remaining LILocBench sequences with the original urllib flow.

Run this file from the LILocBench data directory. Downloads resume directly
into ``*.zip`` files, show the original-style terminal dashboard, retry on
network failures, and validate each completed ZIP. Existing valid archives
are skipped.
"""

import os
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor


BASE_URL = "https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/data/data_files_zipped"

# The five public-GT archives are already present. These are the other 16
# official LILocBench sequences. They are challenge/blind-evaluation data,
# not public pose-label training data.
FILES = {
    f"static_{index}.zip": f"{BASE_URL}/static_{index}.zip"
    for index in range(1, 7)
}
FILES.update({
    f"dynamics_{index}.zip": f"{BASE_URL}/dynamics_{index}.zip"
    for index in range(1, 5)
})
FILES.update({
    f"lt_changes_{index}.zip": f"{BASE_URL}/lt_changes_{index}.zip"
    for index in range(1, 6)
})
FILES["lt_changes_dynamics_1.zip"] = f"{BASE_URL}/lt_changes_dynamics_1.zip"

WORKERS = 3
RETRY_SECONDS = 10
CHUNK_BYTES = 4 * 1024 * 1024

status = {}
lock = threading.Lock()


def human(size):
    if size is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}PB"


def zip_member_count(name):
    try:
        with zipfile.ZipFile(name, "r") as archive:
            return len(archive.infolist())
    except (OSError, zipfile.BadZipFile):
        return None


def set_status(name, downloaded, total, speed, state):
    with lock:
        status[name] = (downloaded, total, speed, state)


def download(name, url):
    count = zip_member_count(name)
    if count is not None:
        size = os.path.getsize(name)
        set_status(name, size, size, 0, f"DONE ({count} files, already present)")
        return

    retries = 0
    while True:
        try:
            existing = os.path.getsize(name) if os.path.exists(name) else 0
            headers = {"User-Agent": "Mozilla/5.0"}
            if existing:
                headers["Range"] = f"bytes={existing}-"
            request = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(request, timeout=60) as response:
                if existing and response.getcode() == 206:
                    mode = "ab"
                    content_range = response.headers.get("Content-Range")
                    total = int(content_range.rsplit("/", 1)[1]) if content_range else None
                else:
                    # The server ignored Range: start safely from byte zero.
                    existing = 0
                    mode = "wb"
                    total = int(response.headers.get("Content-Length", 0)) or None

                downloaded = existing
                started = time.time()
                with open(name, mode) as output:
                    while True:
                        chunk = response.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded += len(chunk)
                        elapsed = max(time.time() - started, 0.001)
                        set_status(name, downloaded, total,
                                   (downloaded - existing) / elapsed, "DOWNLOADING")

            if total is not None and os.path.getsize(name) != total:
                raise RuntimeError(f"size mismatch: {os.path.getsize(name)} != {total}")
            count = zip_member_count(name)
            if count is None:
                raise RuntimeError("ZIP central-directory verification failed")
            set_status(name, os.path.getsize(name), total, 0, f"DONE ({count} files)")
            return
        except Exception as error:
            retries += 1
            set_status(name, os.path.getsize(name) if os.path.exists(name) else 0,
                       None, 0, f"RETRY {retries}: {error}")
            time.sleep(RETRY_SECONDS)


def monitor():
    while True:
        time.sleep(1)
        print("\033[2J\033[H", end="")
        print("=" * 88)
        print("LILocBench remaining-sequence download (resumable)")
        print("=" * 88)
        done = 0
        with lock:
            snapshot = dict(status)
        for name in FILES:
            downloaded, total, speed, state = snapshot[name]
            progress = f"{downloaded / total * 100:6.2f}%" if total else "   ?   "
            print(f"{name:30s} {progress} {human(downloaded):>10s} / "
                  f"{human(total):>10s} {human(speed) + '/s':>12s} {state}")
            if state.startswith("DONE"):
                done += 1
        if done == len(FILES):
            print("\nAll remaining archives passed ZIP verification.")
            return


for filename in FILES:
    status[filename] = (os.path.getsize(filename) if os.path.exists(filename) else 0,
                        None, 0, "STARTING")

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = [pool.submit(download, name, url) for name, url in FILES.items()]
    monitor()
    for future in futures:
        future.result()
