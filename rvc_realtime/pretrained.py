"""训练底模（pretrained G / D）的扫描、导入与解析。

换更好的底模不必改代码，按下面约定放置或从界面导入即可。

目录（按优先级，靠前的同名组合优先）：
  1. assets/pretrained_custom/   用户自行投放，不会提交到仓库
  2. assets/pretrained_v2/       官方 RVC v2 底模
  3. assets/pretrained/          官方 RVC v1 底模
  另外也会扫描 third_party/.../assets/pretrained_v2 与 pretrained，作为缺文件时的回退。

文件名必须成对出现：
  - 推荐（带音高，本工具默认训练用这个）：f0G{32k|40k|48k}.pth 与 f0D{同采样率}.pth
  - 无音高变体也会被列出：G{sr}.pth 与 D{sr}.pth
  - 自定义子目录（例如 assets/pretrained_custom/MyBetterV2/）内可放：
      f0G40k.pth + f0D40k.pth，或 G.pth + D.pth，或任意 *G*.pth 与对应 *D*.pth
    可选 meta.json：{"version":"v2","sample_rate":"40k","label":"显示名"}

界面导入会把生成器（G）和判别器（D）复制到 assets/pretrained_custom/<名称>/，
然后点「刷新底模」即可出现。

默认：RVC v2 40k，即 assets/pretrained_v2/f0G40k.pth 与 f0D40k.pth。
这些是训练起点，不是用户音色。不要把用户 .pth 或他人音色放进仓库。
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rvc_realtime.errors import UsageError
from rvc_realtime.paths import assets_dir, third_party_rvc

PRETRAINED_CUSTOM_DIRNAME = "pretrained_custom"
DEFAULT_VERSION = "v2"
DEFAULT_SAMPLE_RATE = "40k"
_SR_TOKEN = ("32k", "40k", "48k")
_G_D_PAIR = re.compile(
    r"^(?P<pre>.*?)(?P<f0>f0)?G(?P<tag>.*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PretrainedPair:
    key: str
    label: str
    generator: Path
    discriminator: Path
    version: str
    sample_rate: str
    source: str  # custom / official_v2 / official_v1 / bundled
    has_f0: bool

    def exists(self) -> bool:
        return self.generator.is_file() and self.discriminator.is_file()


def pretrained_custom_dir() -> Path:
    return assets_dir() / PRETRAINED_CUSTOM_DIRNAME


def official_pretrained_dir(version: str) -> Path:
    name = "pretrained" if version == "v1" else "pretrained_v2"
    return assets_dir() / name


def ensure_pretrained_custom_dir() -> Path:
    d = pretrained_custom_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _scan_roots() -> list[tuple[Path, str, str]]:
    """(directory, version_hint, source). custom first so user drop-ins win on collision."""
    roots: list[tuple[Path, str, str]] = [
        (pretrained_custom_dir(), "", "custom"),
        (assets_dir() / "pretrained_v2", "v2", "official_v2"),
        (assets_dir() / "pretrained", "v1", "official_v1"),
    ]
    rvc_assets = third_party_rvc() / "assets"
    roots.append((rvc_assets / "pretrained_v2", "v2", "bundled"))
    roots.append((rvc_assets / "pretrained", "v1", "bundled"))
    return roots


def _stem(path: Path) -> str:
    return path.name[:-4] if path.name.lower().endswith(".pth") else path.stem


def _partner_d_stem(g_stem: str) -> Optional[str]:
    m = _G_D_PAIR.match(g_stem)
    if not m:
        return None
    f0 = m.group("f0") or ""
    return f"{m.group('pre')}{f0}D{m.group('tag')}"


def _infer_sr(text: str) -> str:
    lower = text.lower()
    for sr in _SR_TOKEN:
        if sr in lower:
            return sr
    return ""


def _infer_f0(g_stem: str) -> bool:
    return "f0g" in g_stem.lower() or g_stem.lower() in {"f0g", "g"} and "f0" in g_stem.lower()


def _has_f0(g_stem: str) -> bool:
    return bool(re.search(r"f0g", g_stem, re.IGNORECASE))


def _read_meta(folder: Path) -> dict:
    meta = folder / "meta.json"
    if not meta.is_file():
        return {}
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _pair_from_files(
    g: Path,
    d: Path,
    *,
    version: str,
    source: str,
    folder_name: str = "",
    meta: Optional[dict] = None,
) -> Optional[PretrainedPair]:
    if not g.is_file() or not d.is_file():
        return None
    meta = meta or {}
    g_stem = _stem(g)
    sr = str(meta.get("sample_rate") or _infer_sr(g_stem) or _infer_sr(g.parent.name) or "")
    ver = str(meta.get("version") or version or ("v2" if "v2" in str(g).lower() else "") or DEFAULT_VERSION)
    if ver not in {"v1", "v2"}:
        ver = DEFAULT_VERSION
    has_f0 = _has_f0(g_stem) or g.name.lower() in {"g.pth", "generator.pth"}
    if g.name.lower() == "g.pth":
        has_f0 = bool(meta.get("has_f0", True))
    label = str(meta.get("label") or "").strip()
    if not label:
        loc = folder_name or ("自定义" if source == "custom" else f"RVC {ver}")
        kind = "f0" if has_f0 else "no-f0"
        sr_txt = sr or "采样率未知"
        if source.startswith("official") or source == "bundled":
            label = f"RVC {ver} {sr_txt} · {g.name} / {d.name}"
        elif folder_name:
            label = f"自定义 · {folder_name}（{ver} / {sr_txt}，{kind}）"
        else:
            label = f"自定义 · {g.name} / {d.name}（{ver} / {sr_txt}）"
        if ver == DEFAULT_VERSION and sr == DEFAULT_SAMPLE_RATE and has_f0 and source in {"official_v2", "bundled"}:
            label += "（默认）"
    rel = f"{g.parent.as_posix()}|{g.name}|{d.name}"
    key = f"{source}:{ver}:{sr or 'na'}:{'f0' if has_f0 else 'nof0'}:{rel}"
    return PretrainedPair(
        key=key,
        label=label,
        generator=g,
        discriminator=d,
        version=ver,
        sample_rate=sr or DEFAULT_SAMPLE_RATE,
        source=source,
        has_f0=has_f0,
    )


def _collect_in_dir(folder: Path, version_hint: str, source: str, folder_name: str = "") -> list[PretrainedPair]:
    if not folder.is_dir():
        return []
    meta = _read_meta(folder) if folder_name or source == "custom" else {}
    version = str(meta.get("version") or version_hint or "")
    files = {p.name: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pth"}
    pairs: list[PretrainedPair] = []
    used_d: set[str] = set()
    # Prefer explicit f0G*/f0D* then G*/D*
    stems = sorted(files, key=lambda n: (0 if "f0g" in n.lower() else 1, n.lower()))
    for name in stems:
        if "d" in Path(name).stem.lower() and "g" not in Path(name).stem.lower():
            continue
        d_stem = _partner_d_stem(_stem(files[name]))
        if not d_stem:
            continue
        d_name = None
        for cand in files:
            if _stem(files[cand]).lower() == d_stem.lower():
                d_name = cand
                break
        if not d_name or d_name in used_d:
            continue
        used_d.add(d_name)
        pair = _pair_from_files(
            files[name],
            files[d_name],
            version=version,
            source=source,
            folder_name=folder_name,
            meta=meta,
        )
        if pair:
            pairs.append(pair)
    return pairs


def list_pretrained_pairs() -> list[PretrainedPair]:
    """扫描约定目录，返回可选用的 G/D 底模对。同内容去重，自定义优先。"""
    found: list[PretrainedPair] = []
    seen_files: set[tuple[str, str]] = set()
    for root, version_hint, source in _scan_roots():
        if not root.is_dir():
            continue
        for pair in _collect_in_dir(root, version_hint, source, folder_name="" if source != "custom" else ""):
            sig = (str(pair.generator.resolve()), str(pair.discriminator.resolve()))
            if sig in seen_files:
                continue
            seen_files.add(sig)
            found.append(pair)
        if source == "custom" or root.name == PRETRAINED_CUSTOM_DIRNAME:
            for child in sorted(p for p in root.iterdir() if p.is_dir()):
                for pair in _collect_in_dir(child, version_hint, "custom", folder_name=child.name):
                    sig = (str(pair.generator.resolve()), str(pair.discriminator.resolve()))
                    if sig in seen_files:
                        continue
                    seen_files.add(sig)
                    found.append(pair)
    return _sort_pairs(found)


def _sort_pairs(pairs: list[PretrainedPair]) -> list[PretrainedPair]:
    sr_rank = {"": 9, "40k": 0, "48k": 1, "32k": 2}

    def score(p: PretrainedPair) -> tuple:
        official = 0 if p.source in {"official_v2", "bundled"} and p.version == "v2" else (
            1 if p.source in {"official_v1", "bundled"} and p.version == "v1" else 2
        )
        if p.source == "custom":
            official = 3
        default = 0 if (p.version == "v2" and p.sample_rate == "40k" and p.has_f0) else 1
        f0 = 0 if p.has_f0 else 1
        return (official, default, f0, sr_rank.get(p.sample_rate, 8), p.label)

    return sorted(pairs, key=score)


def default_pair(pairs: Optional[list[PretrainedPair]] = None) -> Optional[PretrainedPair]:
    items = pairs if pairs is not None else list_pretrained_pairs()
    for p in items:
        if p.version == DEFAULT_VERSION and p.sample_rate == DEFAULT_SAMPLE_RATE and p.has_f0:
            if p.source in {"official_v2", "bundled", "custom"}:
                return p
    for p in items:
        if p.version == DEFAULT_VERSION and p.sample_rate == DEFAULT_SAMPLE_RATE and p.has_f0:
            return p
    return items[0] if items else None


def pair_choices(pairs: Optional[list[PretrainedPair]] = None) -> tuple[list[str], Optional[str]]:
    items = pairs if pairs is not None else list_pretrained_pairs()
    labels = [p.label for p in items]
    dft = default_pair(items)
    return labels, (dft.label if dft else (labels[0] if labels else None))


def resolve_pretrained(label_or_key: Optional[str], pairs: Optional[list[PretrainedPair]] = None) -> Optional[PretrainedPair]:
    if not label_or_key:
        return default_pair(pairs)
    items = pairs if pairs is not None else list_pretrained_pairs()
    for p in items:
        if p.label == label_or_key or p.key == label_or_key:
            return p
    return default_pair(items)


def _safe_folder_name(name: str) -> str:
    safe = "".join(ch for ch in (name or "").strip() if ch.isalnum() or ch in "-_（）()")
    return safe or "custom_base"


def import_pretrained_pair(
    generator_path: str,
    discriminator_path: str,
    name: Optional[str] = None,
) -> PretrainedPair:
    """把一对 G/D 底模复制到 assets/pretrained_custom/<name>/，刷新后即可选用。"""
    g_src = Path(generator_path).expanduser()
    d_src = Path(discriminator_path).expanduser()
    if not g_src.is_file() or g_src.suffix.lower() != ".pth":
        raise UsageError("请选择有效的生成器（G）底模文件，扩展名应为 .pth。")
    if not d_src.is_file() or d_src.suffix.lower() != ".pth":
        raise UsageError("请选择有效的判别器（D）底模文件，扩展名应为 .pth。")
    folder_name = _safe_folder_name(name or g_src.parent.name or "custom_base")
    dest_dir = ensure_pretrained_custom_dir() / folder_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _dest_name(src: Path, role: str) -> str:
        stem = _stem(src)
        if role == "G" and _partner_d_stem(stem):
            return src.name
        if role == "D":
            return src.name
        return "f0G40k.pth" if role == "G" else "f0D40k.pth"

    # Keep official-style names when possible; otherwise G.pth / D.pth
    g_name = g_src.name if _partner_d_stem(_stem(g_src)) else ("G.pth" if g_src.name.lower() in {"g.pth", "f0g.pth"} else g_src.name)
    d_guess = _partner_d_stem(_stem(g_src))
    if d_guess:
        g_name = g_src.name
        d_name = d_src.name
        # If D filename does not already match, still keep original names so refresh can pair them
        if _stem(d_src).lower() != d_guess.lower():
            # rename D to the expected partner to make pairing robust
            d_name = d_guess + ".pth"
    else:
        g_name = "G.pth"
        d_name = "D.pth"

    g_dest = dest_dir / g_name
    d_dest = dest_dir / d_name
    shutil.copy2(g_src, g_dest)
    shutil.copy2(d_src, d_dest)
    sr = _infer_sr(_stem(g_dest)) or DEFAULT_SAMPLE_RATE
    ver = "v2"
    meta = {
        "version": ver,
        "sample_rate": sr,
        "label": f"自定义 · {folder_name}（{ver} / {sr}）",
        "has_f0": _has_f0(_stem(g_dest)) or g_name.lower() in {"g.pth"},
    }
    (dest_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pair = _pair_from_files(g_dest, d_dest, version=ver, source="custom", folder_name=folder_name, meta=meta)
    if not pair:
        raise UsageError("已复制文件，但无法识别为成对底模。请检查文件名是否为 f0G*.pth / f0D*.pth。")
    return pair


def placement_help(short: bool = False) -> str:
    custom = pretrained_custom_dir()
    v2 = official_pretrained_dir("v2")
    v2_txt = "assets/pretrained_v2" if short else str(v2)
    custom_txt = "assets/pretrained_custom" if short else str(custom)
    return (
        "训练底模约定：把成对的 f0G{32k|40k|48k}.pth 与 f0D{同采样率}.pth "
        f"放到 {v2_txt}（官方 v2）或 {custom_txt}（自定义，可建子目录）。"
        "放入后点「刷新底模」即可选用，不必改代码。默认是 RVC v2 40k（f0G40k / f0D40k）。"
        "这些是训练起点，不要把用户音色放进这里。"
    )
