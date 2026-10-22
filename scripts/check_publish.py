#!/usr/bin/env python3
"""Pre-publication check for this repository.

Fails if the tree contains anything outside the allowlist, build/cache leftovers, oversized
files, media with metadata, or text that looks like a local path, e-mail address or secret.
Optionally writes a manifest with a SHA-256 per allowlisted file.

    python scripts/check_publish.py                 # check only
    python scripts/check_publish.py --manifest publish-manifest.json
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOW = [
    "README.md", "LICENSE", "THIRD_PARTY.md", "pyproject.toml", ".gitignore",
    ".github/workflows/ci.yml",
    "src/cueframe/*.py", "src/cueframe/synth/*.py",
    "examples/*.py", "examples/*.json",
    "scripts/*.py",
    "tests/*.py",
    "docs/*.png", "docs/*.mp4", "docs/*.json", "docs/*.md",
]
# never published; their presence is reported but does not fail the check (they are
# ignored by .gitignore), except egg-info inside src which must be deleted before packaging
JUNK_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv", "build",
             "dist", "work", "out", ".mypy_cache", ".DS_Store"}
MAX_TEXT_BYTES = 200_000
MAX_MEDIA_BYTES = 1_500_000
TEXT_SUFFIXES = {".py", ".md", ".toml", ".json", ".yml", ".yaml", ".txt", ".gitignore", ""}
PATTERNS = {
    "home path": re.compile(r"(/Users/|/home/|C:\\\\Users)"),
    "e-mail": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "token-like": re.compile(r"\b(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,})\b"),
    "credential word": re.compile(r"(?i)\b(password|passwd|api[_-]?key|secret[_-]?key)\s*[:=]"),
}
IGNORABLE_MEDIA_TAGS = {"major_brand", "minor_version", "compatible_brands", "encoder",
                        "handler_name", "language", "vendor_id"}


def allowed(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in ALLOW)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_text(path: Path, problems: list[str]) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(text.splitlines(), 1):
        for name, rx in PATTERNS.items():
            if rx.search(line):
                if name == "home path" and "check_publish.py" in path.name:
                    continue  # the pattern definitions themselves
                problems.append(f"{path.relative_to(ROOT)}:{i}: {name} (content redacted)")


def check_png(path: Path, problems: list[str]) -> None:
    from PIL import Image

    with Image.open(path) as img:
        extra = {k for k in img.info if k not in {"dpi", "gamma", "transparency", "srgb"}}
        text = getattr(img, "text", {})
        if extra or text:
            problems.append(f"{path.relative_to(ROOT)}: PNG metadata {sorted(extra | set(text))}")


def check_mp4(path: Path, problems: list[str]) -> None:
    argv = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
            str(path)]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=True)  # noqa: S603
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        problems.append(f"{path.relative_to(ROOT)}: ffprobe failed: {exc}")
        return
    doc = json.loads(proc.stdout)
    tags = dict(doc.get("format", {}).get("tags", {}))
    for s in doc.get("streams", []):
        tags.update(s.get("tags", {}))
    leftover = {k: v for k, v in tags.items() if k not in IGNORABLE_MEDIA_TAGS}
    if leftover:
        problems.append(f"{path.relative_to(ROOT)}: media metadata keys {sorted(leftover)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, help="write a JSON inventory with SHA-256 hashes")
    ap.add_argument("--strict-clean", action="store_true", help="also reject ignored artifacts")
    args = ap.parse_args()
    if args.manifest and args.manifest.resolve().is_relative_to(ROOT.resolve()):
        ap.error("write the manifest outside the publication tree")

    problems: list[str] = []
    junk: list[str] = []
    files: list[Path] = []
    for path in sorted(ROOT.rglob("*")):
        rel_parts = path.relative_to(ROOT).parts
        if path.is_symlink():
            problems.append(f"{path.relative_to(ROOT)}: symlink is not publishable")
            continue
        if any(part in JUNK_DIRS or part.endswith(".egg-info") for part in rel_parts):
            if path.name in JUNK_DIRS or path.name.endswith(".egg-info"):
                junk.append(str(path.relative_to(ROOT)))
            continue
        if path.is_dir():
            continue
        files.append(path)

    inventory = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if not allowed(rel):
            problems.append(f"{rel}: not in the publish allowlist")
            continue
        size = path.stat().st_size
        if path.suffix in (".png", ".mp4"):
            if size > MAX_MEDIA_BYTES:
                problems.append(f"{rel}: {size} bytes exceeds media limit {MAX_MEDIA_BYTES}")
            (check_png if path.suffix == ".png" else check_mp4)(path, problems)
        else:
            if size > MAX_TEXT_BYTES:
                problems.append(f"{rel}: {size} bytes exceeds text limit {MAX_TEXT_BYTES}")
            if path.suffix in TEXT_SUFFIXES or path.name == ".gitignore":
                scan_text(path, problems)
        inventory.append({"path": rel, "bytes": size, "sha256": sha256(path)})

    for j in junk:
        print(f"junk: {j} (excluded from payload)")
        if args.strict_clean:
            problems.append(f"{j}: artifact inside the clean publication tree")

    for p in problems:
        print("problem:", p)
    print(f"{len(inventory)} allowlisted files, {len(problems)} problems")
    if args.manifest:
        doc = {"root": "cueframe", "files": inventory, "problems": problems, "junk_present": junk}
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"manifest written to {args.manifest}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
