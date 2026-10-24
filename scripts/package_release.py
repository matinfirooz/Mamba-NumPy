"""Build a GitHub-ready ZIP with per-file SHA-256 checksums (stdlib only)."""

import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def included(path):
    relative = path.relative_to(ROOT)
    excluded = {".git", ".venv", "venv", "__pycache__", "runs", "dist", "build",
                ".pytest_cache", ".ipynb_checkpoints"}
    return (path.is_file() and not path.is_symlink()
            and not any(part in excluded or part.endswith(".egg-info") for part in relative.parts)
            and path.suffix not in (".pyc", ".pyo", ".tmp", ".zip")
            and relative.as_posix() != "MANIFEST.sha256")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "mamba-numpy-github.zip")
    args = parser.parse_args()
    files = sorted(path for path in ROOT.rglob("*") if included(path))
    manifest = "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}\n"
                       for path in files)
    manifest_path = ROOT / "MANIFEST.sha256"
    manifest_path.write_text(manifest, encoding="utf-8")
    files.append(manifest_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            info = zipfile.ZipInfo("mamba-numpy/" + path.relative_to(ROOT).as_posix(),
                                   date_time=(2026, 10, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC check failed")
        for line in manifest.splitlines():
            expected, name = line.split("  ", 1)
            actual = hashlib.sha256(archive.read("mamba-numpy/" + name)).hexdigest()
            if actual != expected:
                raise RuntimeError(f"ZIP hash mismatch: {name}")
    print(f"{args.output.resolve()} | {len(files)} files | {args.output.stat().st_size:,} bytes | CRC/SHA-256 verified")


if __name__ == "__main__":
    main()
