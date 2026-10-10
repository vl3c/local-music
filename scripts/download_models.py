"""Fetch the weights listed in models.json into models/<name>/, verifying every file.

Each repo is pinned to a revision in models.json; models.lock.json pins every file's size
and SHA256, so an upstream re-upload fails loudly instead of silently changing what you get.
Downloads resume. Files you already have are reused instead of downloaded: from another
install (--reuse <its models folder>) or from your Hugging Face cache, hard-linked when on
the same drive (no extra space) or copied, then verified like any download. A reused file
may sit under any model folder of the other install: local-sounds keeps the same T5Gemma
text encoder under sa3-small-sfx/, and the checksum decides whether it is the same file.

    python scripts/download_models.py                    # everything not marked optional
    python scripts/download_models.py --only sa3-small-music   # one repo, by name
    python scripts/download_models.py --verify           # check what's there, download nothing
    python scripts/download_models.py --reuse ../local-sounds/models

Gated repos (Stable Audio 3): accept the licence on the model's Hugging Face page and run
`hf auth login` yourself first. This script never reads, prints or stores your token: the
huggingface_hub library finds the login on its own.

Runs with the main venv's Python (it needs huggingface_hub).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
MANIFEST = REPO / "models.json"
LOCK = REPO / "models.lock.json"
CHUNK = 1 << 20

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    total, seen, last = path.stat().st_size, 0, 0.0
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
            seen += len(chunk)
            if total > 200 * CHUNK and time.time() - last > 1:
                print(f"\r    hashing {path.name}: {100 * seen / total:5.1f}%", end="", flush=True)
                last = time.time()
    if total > 200 * CHUNK:
        print("\r" + " " * 70 + "\r", end="")
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def load_lock() -> dict:
    try:
        return json.loads(LOCK.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def wanted(entry: dict, path: str) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in entry["files"])


def remote_files(entry: dict) -> tuple[str, list[dict]]:
    """(resolved revision sha, [{path, bytes, sha256|None}]) from the Hub."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(entry["repo"], revision=entry["revision"], files_metadata=True)
    files = []
    for s in info.siblings:
        if not wanted(entry, s.rfilename):
            continue
        lfs = getattr(s, "lfs", None)
        files.append({"path": s.rfilename, "bytes": s.size,
                      "sha256": (lfs.sha256 if lfs is not None else None)})
    return info.sha, files


def candidates(entry: dict, rel: str, reuse: list[Path], any_folder: bool = False) -> list[Path]:
    """Places an identical copy of this file may already be: other installs, the HF cache.

    any_folder: also look under the other install's other model folders. Only for files with
    a pinned checksum: a small config with the same name and size could be a different file.
    """
    out = [r / entry["name"] / rel for r in reuse]
    if any_folder:
        out += [d / rel for r in reuse if r.is_dir() for d in sorted(r.iterdir())
                if d.is_dir() and d.name != entry["name"]]
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")
    snaps = hub / ("models--" + entry["repo"].replace("/", "--")) / "snapshots"
    if snaps.is_dir():
        out += [s / rel for s in snaps.iterdir()]
    return [p for p in out if p.is_file()]


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".reuse")
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
        how = "hard-linked"
    except OSError:
        shutil.copyfile(src, tmp)
        how = "copied"
    tmp.replace(dst)
    return how


def fetch(entry: dict, lock: dict, reuse: list[Path], verify_only: bool) -> bool:
    name, dest = entry["name"], MODELS / entry["name"]
    print(f"\n[{name}] {entry['repo']}  ({entry['licence']})")
    locked = lock.get(name)
    if locked is None:
        if verify_only:
            print("    not in models.lock.json: nothing to verify against")
            return False
        rev, files = remote_files(entry)
        locked = {"repo": entry["repo"], "revision": rev, "files": files}
    ok = True
    for f in locked["files"]:
        path = dest / f["path"]
        if not verify_only and not (path.is_file() and path.stat().st_size == f["bytes"]):
            for src in candidates(entry, f["path"], reuse, any_folder=f["sha256"] is not None):
                if src.stat().st_size == f["bytes"] and (f["sha256"] is None or sha256(src) == f["sha256"]):
                    print(f"    {f['path']}: {link_or_copy(src, path)} from {src.parent}")
                    break
        if not path.is_file() or path.stat().st_size != f["bytes"]:
            if verify_only:
                print(f"    MISSING {f['path']}")
                ok = False
                continue
            print(f"    {f['path']} ({human(f['bytes'])})", flush=True)
            from huggingface_hub import hf_hub_download
            hf_hub_download(entry["repo"], f["path"], revision=locked["revision"], local_dir=dest)
        if path.stat().st_size != f["bytes"]:
            print(f"    SIZE MISMATCH {f['path']}: {path.stat().st_size} != {f['bytes']}")
            ok = False
            continue
        if f["sha256"] is None:  # small file: pin its hash the first time we see it
            f["sha256"] = sha256(path)
        elif sha256(path) != f["sha256"]:
            print(f"    CHECKSUM MISMATCH {f['path']}: delete it and re-run")
            ok = False
    if ok:
        lock[name] = locked
        total = sum(f["bytes"] for f in locked["files"])
        print(f"    ok: {len(locked['files'])} files, {human(total)}, revision {locked['revision'][:10]}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", help="repo name(s) from models.json (repeatable)")
    ap.add_argument("--verify", action="store_true", help="check files against models.lock.json; download nothing")
    ap.add_argument("--reuse", action="append", type=Path, default=[], help="another install's models folder")
    ap.add_argument("--with-optional", action="store_true", help="include optional repos (Small-Music)")
    args = ap.parse_args()

    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["repos"]
    names = {e["name"] for e in entries}
    if args.only and (bad := set(args.only) - names):
        ap.error(f"unknown repo name(s) {sorted(bad)}; choose from {sorted(names)}")
    todo = [e for e in entries if (e["name"] in args.only if args.only else
                                   (args.with_optional or not e.get("optional")))]
    lock = load_lock()
    failed = []
    for e in todo:
        try:
            if not fetch(e, lock, [r.resolve() for r in args.reuse], args.verify):
                failed.append(e["name"])
        except Exception as exc:  # gated without access, network down...
            msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            if "gated" in msg.lower() or "401" in msg or "403" in msg:
                msg += ("\n    This repo is gated: accept its licence on its Hugging Face page, run "
                        "`hf auth login` yourself, then re-run.")
            print(f"    FAILED: {type(exc).__name__}: {msg}")
            failed.append(e["name"])
        if not args.verify:
            LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    if failed:
        print(f"\nNot complete: {', '.join(failed)}. Re-run to resume.")
        return 1
    print("\nAll weights present and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
