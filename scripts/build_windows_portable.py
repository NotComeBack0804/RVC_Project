#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble a portable Windows DirectML runtime for rvc-realtime-voice.

Runs on Linux or Windows. Downloads official Windows wheels / models, unpacks
an embeddable CPython 3.12 tree, and writes zip archive(s) outside the git
tree. Environment markers are evaluated as Windows so PyTorch's Linux NVIDIA
extras are never bundled. This script cannot execute python.exe or probe a
DirectML adapter when running on Linux.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
REQ_FILE = Path(__file__).resolve().parent / "requirements-windows-directml.txt"
PORTABLE_DIR = Path(__file__).resolve().parent / "windows_portable"

PYTHON_VERSION = "3.12.10"
PYTHON_EMBED_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/"
    f"python-{PYTHON_VERSION}-embed-amd64.zip"
)
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.1-essentials_build.zip"
VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
HF_REPO = "lj1995/VoiceConversionWebUI"
INDEX_URLS = [
    "https://pypi.org/simple",
    "https://download.pytorch.org/whl/cpu",
]
PACKAGE_DIRNAME = "RVC-Realtime-Windows-DirectML"
# Split if a single zip would exceed this many bytes (~1.9 GiB) so Explorer / some
# file hosts can still handle it. Two zips extract into the same folder name.
SPLIT_THRESHOLD = 1900 * 1024 * 1024
BLOCKED_PYPI_PREFIXES = (
    "nvidia-",
    "nvidia_",
    "triton",
    "cuda-python",
)

WIN_MARKER_ENV = {
    "implementation_name": "cpython",
    "implementation_version": PYTHON_VERSION,
    "os_name": "nt",
    "platform_machine": "AMD64",
    "platform_python_implementation": "CPython",
    "platform_release": "10",
    "platform_system": "Windows",
    "platform_version": "10",
    "python_full_version": PYTHON_VERSION,
    "python_version": "3.12",
    "sys_platform": "win32",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def rel(path: Path, start: Path) -> str:
    return path.resolve().relative_to(start.resolve()).as_posix()


def download(url: str, dest: Path, retries: int = 5) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        log(f"  cache hit: {dest.name} ({dest.stat().st_size} bytes)")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            log(f"  downloading {url} -> {dest.name} (try {attempt})")
            req = Request(url, headers={"User-Agent": "rvc-windows-portable-packager"})
            with urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
                shutil.copyfileobj(resp, fh, 1024 * 1024)
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if tmp.exists():
                tmp.unlink()
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed to download {url}: {last_err}") from last_err


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def pip_spec(req: Requirement) -> str:
    """Requirement string pip accepts (no environment marker)."""
    s = req.name
    if req.extras:
        s += "[" + ",".join(req.extras) + "]"
    if req.specifier:
        s += str(req.specifier)
    return s


def parse_root_requirements(path: Path) -> list[str]:
    reqs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        reqs.append(line)
    reqs.append("websockets>=10.4,<11")
    return reqs


def marker_applies(req: Requirement, extra: str = "") -> bool:
    if req.marker is None:
        return True
    env = dict(WIN_MARKER_ENV)
    env["extra"] = extra
    try:
        return bool(req.marker.evaluate(env))
    except Exception:
        return False


def is_blocked(name: str) -> bool:
    n = canonicalize_name(name)
    return n.startswith(BLOCKED_PYPI_PREFIXES) or n in {"triton"}


# ---------------------------------------------------------------------------
# Wheel / sdist cache
# ---------------------------------------------------------------------------

class Artifact:
    def __init__(self, path: Path, name: str, version: str, kind: str):
        self.path = path
        self.name = canonicalize_name(name)
        self.version = Version(version)
        self.kind = kind  # "wheel" | "sdist"


def _dist_from_filename(path: Path) -> tuple[str, str, str] | None:
    name = path.name
    if name.endswith(".whl"):
        # {dist}-{ver}(-{build})?-{py}-{abi}-{plat}.whl
        parts = name[:-4].split("-")
        if len(parts) >= 5:
            # version may contain +cpu; build tag optional
            # We use METADATA later if needed.
            dist = parts[0]
            ver = parts[1]
            return dist, ver, "wheel"
    for suf in (".tar.gz", ".zip"):
        if name.endswith(suf):
            stem = name[: -len(suf)]
            if "-" in stem:
                dist, ver = stem.split("-", 1)
                return dist, ver, "sdist"
    return None


def read_wheel_metadata(path: Path) -> tuple[str, str, list[str]]:
    with zipfile.ZipFile(path) as zf:
        metas = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA") and n.count("/") == 1]
        if not metas:
            raise RuntimeError(f"no METADATA in {path.name}")
        text = zf.read(metas[0]).decode("utf-8", "replace")
    name = version = ""
    requires: list[str] = []
    for line in text.splitlines():
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()
        elif line.startswith("Requires-Dist:"):
            requires.append(line.split(":", 1)[1].strip())
        elif line.strip() == "":
            break
    if not name or not version:
        raise RuntimeError(f"incomplete METADATA in {path.name}")
    return name, version, requires


def read_sdist_requires(path: Path) -> tuple[str, str, list[str]]:
    opener = tarfile.open if path.name.endswith(".tar.gz") else zipfile.ZipFile
    requires: list[str] = []
    name = version = ""
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.name.endswith("PKG-INFO") and m.name.count("/") <= 2]
            if not members:
                raise RuntimeError(f"no PKG-INFO in {path.name}")
            text = tf.extractfile(members[0]).read().decode("utf-8", "replace")  # type: ignore[union-attr]
    else:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.endswith("PKG-INFO") and n.count("/") <= 2]
            text = zf.read(names[0]).decode("utf-8", "replace")
    for line in text.splitlines():
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()
        elif line.startswith("Requires-Dist:"):
            requires.append(line.split(":", 1)[1].strip())
    return name, version, requires


class Inventory:
    def __init__(self) -> None:
        self.by_name: dict[str, Artifact] = {}
        self.requires: dict[str, list[str]] = {}

    def add(self, art: Artifact, requires: list[str]) -> None:
        self.by_name[art.name] = art
        self.requires[art.name] = requires

    def satisfies(self, req: Requirement) -> bool:
        art = self.by_name.get(canonicalize_name(req.name))
        if art is None:
            return False
        if req.specifier and art.version not in req.specifier:
            return False
        return True


def pip_cmd() -> list[str]:
    # Prefer the build venv if present, else current interpreter.
    candidates = [
        Path("/workspace/rvc-windows-cache/build-venv/bin/pip"),
        Path(sys.executable).with_name("pip"),
    ]
    for c in candidates:
        if c.is_file():
            return [str(c)]
    return [sys.executable, "-m", "pip"]


def pip_download(req: str, dest: Path, *, binary: bool) -> Path | None:
    dest.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in dest.iterdir() if p.is_file()}
    try:
        req = pip_spec(Requirement(req))
    except Exception:
        pass
    cmd = pip_cmd() + [
        "download",
        "--no-deps",
        "--dest",
        str(dest),
        "--python-version",
        "312",
        "--implementation",
        "cp",
        "--abi",
        "cp312",
        "--disable-pip-version-check",
        "--exists-action",
        "i",
    ]
    if binary:
        cmd += ["--platform", "win_amd64", "--only-binary", ":all:"]
    else:
        cmd += ["--no-binary", ":all:"]
    for url in INDEX_URLS[1:]:
        cmd += ["--extra-index-url", url]
    cmd.append(req)
    log("  " + " ".join(cmd[-8:]))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        log(proc.stdout[-2000:] if proc.stdout else "pip failed")
        return None
    after = [p for p in dest.iterdir() if p.is_file() and p.name not in before]
    if after:
        after.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return after[0]
    # pip may reuse a file already in dest ("Using cached" / "Saved ...").
    saved = re.findall(r"Saved\s+(\S+)", proc.stdout or "")
    for item in saved:
        cand = Path(item)
        if not cand.is_absolute():
            cand = dest / cand.name
        if cand.is_file():
            return cand
    # Last resort: any archive in dest whose name starts with the project name.
    proj = req.split("==")[0].split(">=")[0].split("<")[0].split("[")[0].strip().replace("-", "_").lower()
    matches = [
        p
        for p in dest.iterdir()
        if p.is_file()
        and p.name.lower().replace("-", "_").startswith(proj)
        and (p.suffix == ".whl" or p.name.endswith(".tar.gz") or p.suffix == ".zip")
    ]
    if matches:
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]
    return None


def _seed_artifact(inv: Inventory, path: Path) -> None:
    if not path.is_file():
        return
    try:
        if path.suffix == ".whl":
            name, ver, reqs = read_wheel_metadata(path)
            inv.add(Artifact(path, name, ver, "wheel"), reqs)
        elif path.name.endswith(".tar.gz") or path.suffix == ".zip":
            if path.name.startswith("python-") or "ffmpeg" in path.name.lower():
                return
            try:
                name, ver, reqs = read_sdist_requires(path)
            except Exception:
                return
            inv.add(Artifact(path, name, ver, "sdist"), reqs)
    except Exception as exc:  # noqa: BLE001
        log(f"  skip cache file {path.name}: {exc}")


def resolve_windows_artifacts(root_reqs: list[str], dest: Path) -> Inventory:
    inv = Inventory()
    sdest = dest.parent / "sdist"
    sdest.mkdir(parents=True, exist_ok=True)
    # Seed from anything already in the wheel / sdist caches.
    for folder in (dest, sdest):
        if not folder.is_dir():
            continue
        for path in folder.glob("*"):
            _seed_artifact(inv, path)

    root_constraints: dict[str, object] = {}
    for raw in root_reqs:
        try:
            r = Requirement(raw)
        except Exception:
            continue
        if r.specifier:
            root_constraints[canonicalize_name(r.name)] = r.specifier

    queue: list[str] = list(root_reqs)
    seen_req: set[str] = set()
    while queue:
        raw = queue.pop(0)
        key = raw.strip()
        if key in seen_req:
            continue
        seen_req.add(key)
        try:
            req = Requirement(key)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"bad requirement {key!r}: {exc}") from exc
        if not marker_applies(req):
            log(f"  skip (Windows marker): {key}")
            continue
        if is_blocked(req.name):
            log(f"  skip blocked (CUDA/NVIDIA): {req.name}")
            continue
        cname = canonicalize_name(req.name)
        if inv.satisfies(req):
            art = inv.by_name[cname]
            for dep in art_requires_as_queue(inv.requires[art.name]):
                queue.append(dep)
            continue
        if cname in inv.by_name:
            art = inv.by_name[cname]
            root_spec = root_constraints.get(cname)
            if root_spec is not None and art.version in root_spec:
                log(f"  keep {art.name}=={art.version} (root pin), skip conflicting {key}")
                continue
        log(f"fetch {key}")
        path = pip_download(key, dest, binary=True)
        if path is None:
            log(f"  no win_amd64 wheel, trying sdist for {key}")
            # sdists live beside wheels; use a sibling folder so ffmpeg zips stay clean
            sdest = dest.parent / "sdist"
            path = pip_download(key, sdest, binary=False)
            if path is None:
                raise RuntimeError(f"could not download Windows artifact for {key}")
        if path.suffix == ".whl":
            name, ver, reqs = read_wheel_metadata(path)
            inv.add(Artifact(path, name, ver, "wheel"), reqs)
        else:
            name, ver, reqs = read_sdist_requires(path)
            inv.add(Artifact(path, name, ver, "sdist"), reqs)
        for dep in art_requires_as_queue(reqs):
            queue.append(dep)
    return inv


def art_requires_as_queue(requires: list[str]) -> list[str]:
    out: list[str] = []
    for item in requires:
        try:
            req = Requirement(item)
        except Exception:
            continue
        # Skip extra-only deps unless they also have a platform marker we want.
        extra = ""
        if req.marker is not None and "extra" in str(req.marker):
            # If the marker is solely an extra (or extra and something false), skip default.
            env = dict(WIN_MARKER_ENV)
            env["extra"] = ""
            try:
                if not req.marker.evaluate(env):
                    continue
            except Exception:
                continue
        if not marker_applies(req, extra=extra):
            continue
        if is_blocked(req.name):
            continue
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Install into embeddable Python
# ---------------------------------------------------------------------------

def install_wheel(wheel: Path, site: Path, scripts: Path, prefix: Path) -> None:
    site.mkdir(parents=True, exist_ok=True)
    scripts.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(site)
    for data_dir in list(site.glob("*.data")):
        # PEP 427 data directories
        for scheme, dest in (
            ("purelib", site),
            ("platlib", site),
            ("scripts", scripts),
            ("data", prefix),
            ("headers", prefix / "Include"),
        ):
            src = data_dir / scheme
            if not src.exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            for item in src.rglob("*"):
                if item.is_file():
                    target = dest / item.relative_to(src)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
        shutil.rmtree(data_dir, ignore_errors=True)


def install_websockets_sdist(sdist: Path, site: Path) -> None:
    """Copy the pure-Python package; skip compiling Windows speedups on Linux."""
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(sdist, "r:gz") as tf:
            tf.extractall(td)
        roots = [p for p in Path(td).iterdir() if p.is_dir()]
        src_root = roots[0]
        pkg = None
        for cand in (src_root / "src" / "websockets", src_root / "websockets"):
            if cand.is_dir():
                pkg = cand
                break
        if pkg is None:
            raise RuntimeError("websockets sdist layout not recognized")
        dest = site / "websockets"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(pkg, dest, ignore=shutil.ignore_patterns("*.c", "*.h", "*.so", "*.pyd"))
        # Minimal dist-info so importlib.metadata works.
        info = site / "websockets-10.4.dist-info"
        info.mkdir(exist_ok=True)
        (info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: websockets\nVersion: 10.4\n",
            encoding="utf-8",
        )
        (info / "INSTALLER").write_text("rvc-windows-portable\n", encoding="utf-8")


def install_generic_sdist(sdist: Path, site: Path) -> None:
    """Install a pure-Python sdist by copying its package dir(s)."""
    if "websockets" in sdist.name.lower():
        install_websockets_sdist(sdist, site)
        return
    with tempfile.TemporaryDirectory() as td:
        if sdist.name.endswith(".tar.gz"):
            with tarfile.open(sdist, "r:gz") as tf:
                tf.extractall(td)
        else:
            with zipfile.ZipFile(sdist) as zf:
                zf.extractall(td)
        roots = [p for p in Path(td).iterdir() if p.is_dir()]
        src_root = roots[0]
        # Refuse obvious native extension sdists (except known-optional C files).
        native = [
            p
            for p in src_root.rglob("*")
            if p.suffix in {".c", ".cpp", ".cu", ".rs"} and "test" not in p.as_posix().lower()
        ]
        # setuptools / packages: copy top-level packages and modules.
        candidates: list[Path] = []
        for base in (src_root, src_root / "src"):
            if not base.is_dir():
                continue
            for item in base.iterdir():
                if item.name in {"tests", "test", "docs", "examples", "src"}:
                    continue
                if item.is_dir() and ((item / "__init__.py").exists() or (item / "__init__.pyi").exists()):
                    candidates.append(item)
                elif item.suffix == ".py" and item.name not in {"setup.py", "setup.cfg", "conf.py"}:
                    candidates.append(item)
        if not candidates:
            raise RuntimeError(f"cannot install sdist {sdist.name}: no pure-Python package found")
        if native and not any("websockets" in sdist.name.lower() for _ in [0]):
            # Still allow if we only copy .py and the extension is optional — but warn.
            log(f"  warning: {sdist.name} contains native sources; copying Python files only")
        for item in candidates:
            dest = site / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest, ignore=shutil.ignore_patterns("*.c", "*.cpp", "*.h", "*.so", "*.pyd"))
            else:
                shutil.copy2(item, dest)


def configure_pth(runtime: Path) -> None:
    pths = list(runtime.glob("python*._pth"))
    if not pths:
        raise RuntimeError("embeddable Python is missing pythonXX._pth")
    pth = pths[0]
    pth.write_text(
        "python312.zip\n.\nLib\\site-packages\n..\n"
        "..\\third_party\\Retrieval-based-Voice-Conversion-WebUI\n"
        "import site\n",
        encoding="utf-8",
    )


def unpack_embeddable_python(zip_path: Path, runtime: Path) -> None:
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(runtime)
    configure_pth(runtime)
    (runtime / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    (runtime / "Scripts").mkdir(parents=True, exist_ok=True)


def unpack_ffmpeg(zip_path: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(td)
        # find bin/ffmpeg.exe
        bins = list(Path(td).rglob("ffmpeg.exe"))
        if not bins:
            raise RuntimeError("ffmpeg.exe not found in essentials zip")
        bin_dir = bins[0].parent
        for item in bin_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dest / item.name)
        # copy licenses sitting beside bin or at tree root
        for lic in Path(td).rglob("*"):
            if not lic.is_file():
                continue
            lower = lic.name.lower()
            if lower.startswith("license") or lower.startswith("copying") or lower in {"readme.txt", "readme.md"}:
                shutil.copy2(lic, dest / lic.name)


# ---------------------------------------------------------------------------
# Copy project sources into the stage
# ---------------------------------------------------------------------------

IGNORE_NAMES = {
    ".git",
    ".github",
    ".gitmodules",
    "__pycache__",
    ".venv",
    ".venv-rvc",
    ".model-downloads",
    ".cache",
    "RVCRealtimeVST",
}


def ignore_copy(dirpath: str, names: list[str]) -> set[str]:
    ignored = set()
    for n in names:
        if n in IGNORE_NAMES or n.endswith(".pyc") or n.endswith(".pyo"):
            ignored.add(n)
            continue
        if n.startswith(".") and n not in {".gitkeep"}:
            # keep .gitkeep; drop other dotfiles/dirs like .cache
            if n not in {".gitkeep"}:
                ignored.add(n)
    return ignored


def copy_project_sources(src_root: Path, dest_root: Path) -> None:
    # rvc_realtime package
    shutil.copytree(
        src_root / "rvc_realtime",
        dest_root / "rvc_realtime",
        ignore=ignore_copy,
        dirs_exist_ok=True,
    )
    for name in ("README.md", "requirements-lite.txt"):
        p = src_root / name
        if p.exists():
            shutil.copy2(p, dest_root / name)
    # scripts (packaging helpers included, useful for optional model download)
    shutil.copytree(
        src_root / "scripts",
        dest_root / "scripts",
        ignore=ignore_copy,
        dirs_exist_ok=True,
    )
    # docs + tools
    if (src_root / "docs").is_dir():
        shutil.copytree(src_root / "docs", dest_root / "docs", ignore=ignore_copy, dirs_exist_ok=True)
    if (src_root / "tools").is_dir():
        shutil.copytree(src_root / "tools", dest_root / "tools", ignore=ignore_copy, dirs_exist_ok=True)
    # testdata placeholders only
    for sub, keep in (
        ("authorized_reference", {".gitkeep", "README.md"}),
        ("recorded", {".gitkeep"}),
        ("generated", {".gitkeep"}),
    ):
        dest = dest_root / "testdata" / sub
        dest.mkdir(parents=True, exist_ok=True)
        src = src_root / "testdata" / sub
        if src.is_dir():
            for item in src.iterdir():
                if item.name in keep and item.is_file():
                    shutil.copy2(item, dest / item.name)
    # official RVC sources, without logs (except mute) and without asset symlinks
    rvc_src = src_root / "third_party" / "Retrieval-based-Voice-Conversion-WebUI"
    rvc_dst = dest_root / "third_party" / "Retrieval-based-Voice-Conversion-WebUI"
    if not rvc_src.is_dir():
        raise RuntimeError(
            "official RVC checkout missing; clone "
            "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI "
            "into third_party/Retrieval-based-Voice-Conversion-WebUI"
        )

    def ignore_rvc(dirpath: str, names: list[str]) -> set[str]:
        ignored = ignore_copy(dirpath, names)
        rel = Path(dirpath).resolve().relative_to(rvc_src.resolve()).as_posix()
        if rel in {".", ""}:
            ignored.update({"logs", "assets", "RVCRealtimeVST"})
        return ignored

    shutil.copytree(rvc_src, rvc_dst, ignore=ignore_rvc, dirs_exist_ok=True)
    # pymss yaml configs
    pymss = rvc_src / "assets" / "pymss_weights"
    if pymss.is_dir():
        dst = rvc_dst / "assets" / "pymss_weights"
        dst.mkdir(parents=True, exist_ok=True)
        for item in pymss.iterdir():
            if item.is_file() and not item.is_symlink():
                shutil.copy2(item, dst / item.name)
    (rvc_dst / "assets").mkdir(parents=True, exist_ok=True)
    # mute samples only
    mute_src = rvc_src / "logs" / "mute"
    if not mute_src.is_dir():
        raise RuntimeError("official mute samples missing; run download_base_models.py --with-pretrained once")
    shutil.copytree(mute_src, rvc_dst / "logs" / "mute", ignore=ignore_copy)
    # third_party README
    tp_readme = src_root / "third_party" / "README.md"
    if tp_readme.is_file():
        (dest_root / "third_party").mkdir(exist_ok=True)
        shutil.copy2(tp_readme, dest_root / "third_party" / "README.md")


def copy_base_models(src_root: Path, dest_root: Path, cache: Path) -> None:
    mapping = [
        (src_root / "assets" / "hubert_base" / "config.json", dest_root / "assets" / "hubert_base" / "config.json"),
        (src_root / "assets" / "hubert_base" / "preprocessor_config.json", dest_root / "assets" / "hubert_base" / "preprocessor_config.json"),
        (src_root / "assets" / "hubert_base" / "pytorch_model.bin", dest_root / "assets" / "hubert_base" / "pytorch_model.bin"),
        (src_root / "assets" / "rmvpe" / "rmvpe.pt", dest_root / "assets" / "rmvpe" / "rmvpe.pt"),
        (src_root / "assets" / "pretrained_v2" / "f0G40k.pth", dest_root / "assets" / "pretrained_v2" / "f0G40k.pth"),
        (src_root / "assets" / "pretrained_v2" / "f0D40k.pth", dest_root / "assets" / "pretrained_v2" / "f0D40k.pth"),
    ]
    onnx_src = src_root / "assets" / "rmvpe" / "rmvpe.onnx"
    onnx_cache = cache / "models" / "rmvpe.onnx"
    if onnx_src.is_file():
        mapping.append((onnx_src, dest_root / "assets" / "rmvpe" / "rmvpe.onnx"))
    elif onnx_cache.is_file():
        mapping.append((onnx_cache, dest_root / "assets" / "rmvpe" / "rmvpe.onnx"))
    else:
        raise RuntimeError("rmvpe.onnx is missing; download it into the cache first")
    for src, dest in mapping:
        if not src.is_file() or src.stat().st_size == 0:
            raise RuntimeError(f"required model missing: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        log(f"  model {src.name} ({src.stat().st_size} bytes)")
        shutil.copy2(src, dest)
    for name in ("weights", "indices"):
        d = dest_root / "assets" / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").write_text("", encoding="utf-8")
    custom = dest_root / "assets" / "pretrained_custom"
    if custom.exists():
        shutil.rmtree(custom)
    custom.mkdir(parents=True, exist_ok=True)
    (custom / ".gitkeep").write_text("", encoding="utf-8")
    readme_src = src_root / "assets" / "pretrained_custom" / "README.md"
    if readme_src.is_file():
        shutil.copy2(readme_src, custom / "README.md")
    # do not ship leftover user / smoke models


def copy_launchers(dest_root: Path, runtime: Path) -> None:
    files = {
        "Start-UI.bat": dest_root / "Start-UI.bat",
        "Check-GPU.bat": dest_root / "Check-GPU.bat",
        "Start-UI-CPU.bat": dest_root / "Start-UI-CPU.bat",
        "启动界面.bat": dest_root / "启动界面.bat",
        "查看计算设备.bat": dest_root / "查看计算设备.bat",
        "使用CPU启动.bat": dest_root / "使用CPU启动.bat",
        "README.txt": dest_root / "README.txt",
        "NOTICE.txt": dest_root / "NOTICE.txt",
        "sitecustomize.py": runtime / "sitecustomize.py",
    }
    for name, dest in files.items():
        src = PORTABLE_DIR / name
        if not src.is_file():
            raise RuntimeError(f"missing packager file {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    exe_src = PORTABLE_DIR / "Start-UI.exe"
    if not exe_src.is_file():
        maybe_compile_launcher(exe_src)
    if exe_src.is_file():
        shutil.copy2(exe_src, dest_root / "Start-UI.exe")
        shutil.copy2(exe_src, dest_root / "启动界面.exe")


def maybe_compile_launcher(exe: Path) -> None:
    src = PORTABLE_DIR / "launcher.c"
    cc = shutil.which("x86_64-w64-mingw32-gcc")
    if not cc or not src.is_file():
        return
    log("compiling Start-UI.exe")
    subprocess.run([cc, "-O2", "-municode", "-o", str(exe), str(src)], check=True)


def write_root_docs(dest_root: Path) -> None:
    # UTF-8 BOM copy of the short guide also as 使用说明.txt
    readme = dest_root / "README.txt"
    if readme.is_file():
        data = readme.read_bytes()
        (dest_root / "使用说明.txt").write_bytes(data)
    # ensure docs pointer exists
    if not (dest_root / "docs" / "windows-portable.md").is_file():
        raise RuntimeError("docs/windows-portable.md was not copied")


# ---------------------------------------------------------------------------
# Verification (Linux-safe)
# ---------------------------------------------------------------------------

def must_exist(path: Path, min_size: int = 1) -> None:
    if not path.exists():
        raise RuntimeError(f"missing: {path}")
    if path.is_file() and path.stat().st_size < min_size:
        raise RuntimeError(f"too small ({path.stat().st_size} < {min_size}): {path}")


def verify_stage(dest_root: Path) -> list[str]:
    notes: list[str] = []
    runtime = dest_root / "runtime"
    site = runtime / "Lib" / "site-packages"
    must_exist(runtime / "python.exe")
    must_exist(runtime / "python312.dll")
    pth = runtime / "python312._pth"
    must_exist(pth)
    pth_text = pth.read_text(encoding="utf-8")
    if "Lib\\site-packages" not in pth_text and "Lib/site-packages" not in pth_text:
        raise RuntimeError("python312._pth does not include Lib\\site-packages")
    if "import site" not in pth_text:
        raise RuntimeError("python312._pth does not enable site")
    if ".." not in pth_text.splitlines():
        raise RuntimeError("python312._pth does not put the package root on sys.path")
    if "..\\third_party\\Retrieval-based-Voice-Conversion-WebUI" not in pth_text.splitlines():
        raise RuntimeError("python312._pth does not put the official RVC package on sys.path")
    must_exist(runtime / "sitecustomize.py")
    must_exist(runtime / "ffmpeg" / "ffmpeg.exe")
    must_exist(runtime / "ffmpeg" / "ffprobe.exe")
    must_exist(runtime / "vc_redist.x64.exe", 1_000_000)
    must_exist(dest_root / "Start-UI.bat")
    must_exist(dest_root / "启动界面.bat")
    must_exist(dest_root / "Check-GPU.bat")
    must_exist(dest_root / "Start-UI-CPU.bat")
    must_exist(dest_root / "README.txt")
    must_exist(dest_root / "NOTICE.txt")
    must_exist(dest_root / "docs" / "LICENSE-RVC.txt")
    must_exist(dest_root / "third_party" / "Retrieval-based-Voice-Conversion-WebUI" / "LICENSE")
    must_exist(dest_root / "rvc_realtime" / "app.py")
    must_exist(dest_root / "rvc_realtime" / "compute.py")
    must_exist(dest_root / "rvc_realtime" / "pretrained.py")
    must_exist(dest_root / "rvc_realtime" / "ui.css", 1000)
    must_exist(dest_root / "rvc_realtime" / "windows_bootstrap.py")
    must_exist(dest_root / "docs" / "windows-portable.md")
    must_exist(dest_root / "docs" / "pretrained-bases.md")
    must_exist(dest_root / "assets" / "pretrained_custom" / "README.md")
    must_exist(dest_root / "assets" / "hubert_base" / "pytorch_model.bin", 100 * 1024 * 1024)
    must_exist(dest_root / "assets" / "rmvpe" / "rmvpe.pt", 100 * 1024 * 1024)
    must_exist(dest_root / "assets" / "rmvpe" / "rmvpe.onnx", 100 * 1024 * 1024)
    must_exist(dest_root / "assets" / "pretrained_v2" / "f0G40k.pth", 50 * 1024 * 1024)
    must_exist(dest_root / "assets" / "pretrained_v2" / "f0D40k.pth", 100 * 1024 * 1024)
    must_exist(
        dest_root
        / "third_party"
        / "Retrieval-based-Voice-Conversion-WebUI"
        / "logs"
        / "mute"
        / "0_gt_wavs"
        / "mute40k.wav"
    )
    # runtime modules
    must_exist(site / "torch" / "__init__.py")
    must_exist(site / "torch_directml" / "__init__.py")
    # onnxruntime-directml exposes the onnxruntime package
    if not (site / "onnxruntime").is_dir():
        raise RuntimeError("onnxruntime package missing (onnxruntime-directml wheel)")
    dml_hits = list((site / "onnxruntime").rglob("*DirectML*")) + list((site / "onnxruntime").rglob("*directml*"))
    if not dml_hits:
        # still accept if METADATA says directml
        metas = list(site.glob("onnxruntime_directml-*.dist-info"))
        if not metas:
            raise RuntimeError("cannot confirm onnxruntime-directml was installed")
        notes.append("onnxruntime DirectML DLL name not found by glob; dist-info present")
    else:
        notes.append(f"onnxruntime DirectML files: {', '.join(p.name for p in dml_hits[:6])}")
    # must not ship CUDA / NVIDIA wheels or user voices
    nvidia = [p for p in site.glob("*") if p.name.lower().startswith("nvidia")]
    if nvidia:
        raise RuntimeError("NVIDIA/CUDA packages were installed into the portable runtime: " + ", ".join(p.name for p in nvidia))
    if (dest_root / ".git").exists() or (dest_root / "third_party" / "Retrieval-based-Voice-Conversion-WebUI" / ".git").exists():
        raise RuntimeError(".git must not be included")
    if (dest_root / ".venv").exists() or (dest_root / ".venv-rvc").exists():
        raise RuntimeError("venv directories must not be included")
    banned_frags = ("shadowww", "smoke_jfk", "game_voice_smoke", "keruan", "kikiV1")
    for path in list((dest_root / "assets").rglob("*")) + list((dest_root / "testdata").rglob("*")):
        low = path.name.lower()
        if any(b in low for b in banned_frags):
            raise RuntimeError(f"banned user/smoke artifact included: {path}")
        if path.suffix.lower() in {".pth", ".index"} and path.parent.name in {"weights", "indices"}:
            if path.name != ".gitkeep":
                raise RuntimeError(f"user/smoke weight or index included: {path}")
    # launchers mention DirectML / status / VB-CABLE
    bat = (dest_root / "Start-UI.bat").read_text(encoding="utf-8-sig", errors="replace")
    if "rvc_realtime serve" not in bat or "127.0.0.1" not in bat:
        raise RuntimeError("Start-UI.bat does not launch the local UI")
    css = (dest_root / "rvc_realtime" / "ui.css").read_text(encoding="utf-8")
    for needle in ("#settings-modal", "#stage-train", "#step-record:checked", "--text-dim"):
        if needle not in css:
            raise RuntimeError(f"ui.css is missing v4 selector or variable: {needle}")
    readme = (dest_root / "README.txt").read_text(encoding="utf-8-sig", errors="replace")
    if "系统设置" not in readme or "训练底模" not in readme:
        raise RuntimeError("README.txt does not describe the v4 settings modal or replaceable bases")
    notes.append("directory structure and required files look complete (Linux static checks)")
    notes.append("not verified: running python.exe, DirectML adapter probe, audio I/O, VB-CABLE")
    return notes


# ---------------------------------------------------------------------------
# Zip
# ---------------------------------------------------------------------------

def _zipinfo_for(file_path: Path, arcname: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo.from_file(file_path, arcname)
    info.flag_bits |= 0x800  # UTF-8 names
    # Already-compressed media/models: store or light deflate
    if file_path.suffix.lower() in {".pth", ".onnx", ".bin", ".pt", ".whl", ".exe", ".dll", ".pyd", ".zip"}:
        info.compress_type = zipfile.ZIP_DEFLATED
        info._compresslevel = 1  # type: ignore[attr-defined]
    else:
        info.compress_type = zipfile.ZIP_DEFLATED
        info._compresslevel = 6  # type: ignore[attr-defined]
    return info


def add_file_to_zip(zf: zipfile.ZipFile, file_path: Path, arcname: str) -> None:
    info = _zipinfo_for(file_path, arcname)
    with file_path.open("rb") as src, zf.open(info, "w") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)


def iter_stage_files(dest_root: Path, *, models_only: bool | None = None) -> Iterable[tuple[Path, str]]:
    """models_only=True → only large assets; False → everything except those; None → all."""
    model_rel_prefixes = (
        "assets/hubert_base/",
        "assets/rmvpe/",
        "assets/pretrained_v2/",
        "assets/pretrained/",
    )
    for path in dest_root.rglob("*"):
        if not path.is_file():
            continue
        rel_posix = path.relative_to(dest_root).as_posix()
        is_model = rel_posix.startswith(model_rel_prefixes) and path.suffix.lower() in {
            ".bin",
            ".pt",
            ".onnx",
            ".pth",
            ".json",
        }
        # keep tiny json with models zip too so the folder is self-describing; runtime zip
        # still has empty dirs via other files.
        if models_only is True and not is_model:
            continue
        if models_only is False and is_model:
            continue
        arc = f"{PACKAGE_DIRNAME}/{rel_posix}"
        yield path, arc


def write_zip(files: Iterable[tuple[Path, str]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    log(f"writing {out_path}")
    count = 0
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path, arc in files:
            add_file_to_zip(zf, path, arc)
            count += 1
            if count % 200 == 0:
                log(f"  ... {count} files")
    log(f"  {count} files, {out_path.stat().st_size} bytes")
    return out_path


def write_checksums(paths: list[Path], dest: Path) -> None:
    lines = []
    for p in paths:
        digest = sha256_file(p)
        lines.append(f"{digest}  {p.name}")
        log(f"  SHA256 {p.name} = {digest}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def default_cache() -> Path:
    return Path(os.environ.get("RVC_WINDOWS_CACHE", "/workspace/rvc-windows-cache"))


def default_stage() -> Path:
    return Path(os.environ.get("RVC_WINDOWS_STAGE", "/workspace/rvc-windows-stage"))


def default_output() -> Path:
    return Path(os.environ.get("RVC_WINDOWS_OUTPUT", "/workspace/rvc-windows-directml.zip"))


def build(cache: Path, stage: Path, output: Path, src_root: Path, *, reuse_runtime: bool = False) -> list[Path]:
    cache.mkdir(parents=True, exist_ok=True)
    wheels = cache / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    (cache / "models").mkdir(exist_ok=True)
    (cache / "python").mkdir(exist_ok=True)
    (cache / "ffmpeg").mkdir(exist_ok=True)
    (cache / "redist").mkdir(exist_ok=True)

    log("== downloading runtime ingredients ==")
    py_zip = download(PYTHON_EMBED_URL, cache / "python" / f"python-{PYTHON_VERSION}-embed-amd64.zip")
    ff_zip = download(FFMPEG_URL, cache / "ffmpeg" / "ffmpeg-9.1-essentials_build.zip")
    vcredist = download(VC_REDIST_URL, cache / "redist" / "vc_redist.x64.exe")
    onnx_cache = cache / "models" / "rmvpe.onnx"
    if not onnx_cache.is_file():
        # Prefer already-downloaded project/cache file; otherwise use huggingface_hub if present.
        src_onnx = src_root / "assets" / "rmvpe" / "rmvpe.onnx"
        if src_onnx.is_file():
            shutil.copy2(src_onnx, onnx_cache)
        else:
            try:
                from huggingface_hub import hf_hub_download

                p = hf_hub_download(
                    repo_id=HF_REPO,
                    filename="rmvpe.onnx",
                    revision="main",
                    local_dir=str(cache / "models"),
                )
                log(f"  downloaded rmvpe.onnx via huggingface_hub -> {p}")
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"rmvpe.onnx is missing and could not be downloaded: {exc}") from exc

    log("== resolving Windows wheels (markers evaluated as Windows) ==")
    inv = resolve_windows_artifacts(parse_root_requirements(REQ_FILE), wheels)
    log(f"  {len(inv.by_name)} distributions")
    required = {"torch", "torch-directml", "onnxruntime-directml", "gradio", "sounddevice", "numpy"}
    have = set(inv.by_name)
    missing = [n for n in required if n not in have]
    if missing:
        raise RuntimeError(f"missing required distributions: {missing}")

    dest_root = stage / PACKAGE_DIRNAME
    runtime = dest_root / "runtime"
    site = runtime / "Lib" / "site-packages"
    can_reuse = (
        reuse_runtime
        and (runtime / "python.exe").is_file()
        and (runtime / "python312.dll").is_file()
        and (site / "torch").is_dir()
        and (site / "torch_directml").is_dir()
        and (site / "onnxruntime").is_dir()
        and (runtime / "ffmpeg" / "ffmpeg.exe").is_file()
    )
    if can_reuse:
        log("== reusing existing stage runtime (skip unpack / wheel install) ==")
        for name in ("rvc_realtime", "docs", "scripts", "tools", "testdata", "third_party"):
            pth = dest_root / name
            if pth.exists():
                shutil.rmtree(pth)
    else:
        if reuse_runtime:
            log("== --reuse-runtime requested but stage runtime is incomplete; doing a full rebuild ==")
        if stage.exists():
            shutil.rmtree(stage)
        dest_root.mkdir(parents=True)
        runtime = dest_root / "runtime"
        site = runtime / "Lib" / "site-packages"

        log("== unpacking embeddable Python + ffmpeg ==")
        unpack_embeddable_python(py_zip, runtime)
        unpack_ffmpeg(ff_zip, runtime / "ffmpeg")
        shutil.copy2(vcredist, runtime / "vc_redist.x64.exe")

        log("== installing Windows wheels into runtime/Lib/site-packages ==")
        scripts = runtime / "Scripts"
        for art in sorted(inv.by_name.values(), key=lambda a: a.name):
            log(f"  install {art.name}=={art.version} ({art.kind})")
            if art.kind == "wheel":
                install_wheel(art.path, site, scripts, runtime)
            else:
                install_generic_sdist(art.path, site)

        # Some caches may drop raw .whl files into site-packages; they are not importable and waste space.
        for leftover in site.rglob("*.whl"):
            log(f"  removing leftover {leftover.relative_to(site)}")
            leftover.unlink()

    log("== copying project sources and models ==")
    copy_project_sources(src_root, dest_root)
    copy_base_models(src_root, dest_root, cache)
    copy_launchers(dest_root, runtime)
    write_root_docs(dest_root)

    log("== verifying stage ==")
    notes = verify_stage(dest_root)
    for n in notes:
        log("  " + n)

    # Decide single vs split zip by estimating uncompressed model + runtime sizes
    total = sum(p.stat().st_size for p in dest_root.rglob("*") if p.is_file())
    log(f"  staged size (uncompressed): {total} bytes")
    outputs: list[Path] = []
    # Models alone are ~0.9GB; with torch etc. compressed zip may still be >2GB.
    # Prefer a single zip first; if the result exceeds the threshold, rewrite as two zips.
    single = output
    write_zip(iter_stage_files(dest_root, models_only=None), single)
    if single.stat().st_size <= SPLIT_THRESHOLD:
        outputs = [single]
    else:
        log(f"single zip is {single.stat().st_size} bytes, splitting into runtime + models")
        single.unlink()
        runtime_zip = output.with_name(output.stem + "-runtime.zip")
        models_zip = output.with_name(output.stem + "-models.zip")
        write_zip(iter_stage_files(dest_root, models_only=False), runtime_zip)
        write_zip(iter_stage_files(dest_root, models_only=True), models_zip)
        outputs = [runtime_zip, models_zip]
        # short extract note next to the zips
        note = output.with_name(output.stem + "-README.txt")
        note.write_text(
            "请先解压 {0}，再把 {1} 解压到同一个父目录，"
            "得到文件夹 {2} 后双击其中的 Start-UI.bat。\n"
            "两个压缩包必须合在一起使用：第二个包提供 HuBERT / RMVPE / 训练底模。\n".format(
                runtime_zip.name, models_zip.name, PACKAGE_DIRNAME
            ),
            encoding="utf-8",
        )
        outputs.append(note)
    sums = output.with_name(output.stem + "-SHA256SUMS.txt")
    write_checksums([p for p in outputs if p.suffix.lower() == ".zip"], sums)
    outputs.append(sums)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Windows DirectML portable package")
    parser.add_argument("--cache", type=Path, default=default_cache())
    parser.add_argument("--stage", type=Path, default=default_stage())
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--src", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--reuse-runtime",
        action="store_true",
        help="Keep an existing stage runtime (skip unpacking Python and reinstalling wheels). Still recopies current sources, docs, launchers and base models.",
    )
    args = parser.parse_args(argv)
    outputs = build(
        args.cache.resolve(),
        args.stage.resolve(),
        args.output.resolve(),
        args.src.resolve(),
        reuse_runtime=args.reuse_runtime,
    )
    log("== done ==")
    for p in outputs:
        size = p.stat().st_size if p.is_file() else 0
        log(f"  {p} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
