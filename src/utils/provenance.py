"""Stamp result files with enough context to reproduce or distrust them.

On 2026-07-27 a table of accuracy-vs-refresh-period numbers was written into a
report by a script that was never committed. Three days later it could not be
reproduced, the re-measurement disagreed by 1.5pp, and the original numbers had
to be retracted with the cause unexplained -- see
reports/realtime_incremental_result.md.

The fix is that every result file records what produced it. The field that
matters most is `dirty`: a commit SHA identifies the code only if the working
tree was clean, and the 07-27 numbers came from a tree that was not.

    from src.utils.provenance import save_results
    save_results(Path("reports/foo_raw.json"), {"acc": 0.9})

Metadata lands under the `_meta` key alongside the payload's own keys, so
readers that look up keys by name are unaffected. Code that iterates the top
level should skip keys starting with "_", or use `load_results`, which strips it.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

META_KEY = "_meta"


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(("git", *args), capture_output=True, text=True, timeout=10,
                           cwd=Path(__file__).resolve().parents[2])
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "torch", "sklearn", "transformers"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


def provenance(extra: dict | None = None) -> dict:
    """Who/what/when produced this result."""
    status = _git("status", "--porcelain")
    dirty_files = [ln[3:] for ln in status.split("\n") if ln] if status else []
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # True means the commit above does NOT identify the code that ran.
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files[:20],
        "script": sys.argv[0] or None,
        "argv": sys.argv[1:],
        "env": _versions(),
    }
    if extra:
        meta.update(extra)
    return meta


def save_results(path: str | Path, payload: dict, extra_meta: dict | None = None,
                 indent: int = 2) -> Path:
    """Write `payload` as JSON with a `_meta` provenance block added."""
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a dict to carry {META_KEY!r}, got {type(payload).__name__}")
    if META_KEY in payload:
        raise ValueError(f"payload already has a {META_KEY!r} key")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({META_KEY: provenance(extra_meta), **payload}, indent=indent))
    return path


def load_results(path: str | Path) -> tuple[dict, dict | None]:
    """Inverse of `save_results`: returns (payload without _meta, meta or None)."""
    d = json.loads(Path(path).read_text())
    meta = d.get(META_KEY)
    return {k: v for k, v in d.items() if k != META_KEY}, meta


def describe(path: str | Path) -> str:
    """One-line summary for spot-checking a result file's trustworthiness."""
    _, m = load_results(path)
    if not m or m.get("provenance") == "unknown":
        return f"{Path(path).name}: NO PROVENANCE -- cannot be tied to any commit"
    warn = "  !! DIRTY TREE -- commit does not identify this code" if m.get("dirty") else ""
    return (f"{Path(path).name}: {m.get('git_commit', '?')[:8]} "
            f"({m.get('git_branch', '?')}) {m.get('generated_utc', '?')} "
            f"via {m.get('script', '?')}{warn}")
