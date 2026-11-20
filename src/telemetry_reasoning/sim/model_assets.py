"""Restore the Panda subtree from the release-pinned Menagerie revision.

Git blob hashes bind every downloaded byte to the pinned tree. Only Panda is
needed by this project; other Menagerie models are deliberately not downloaded.
This installs verified asset files, not a Git submodule checkout.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen


MENAGERIE_REVISION = "f3b6013629c7b94f8ee93fd977b161fa8350f9bc"
REPOSITORY = "google-deepmind/mujoco_menagerie"
SUBTREE = "franka_emika_panda"
PANDA_INVENTORY_SHA256 = "86365ae032e08870a8459f52e2e0194f7d00a379ade9f9641209747a8d48162b"


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "telemetry-reasoning-dataset-restoration"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def panda_inventory(tree: dict) -> list[dict]:
    """Validate a nontruncated pinned Git tree before accepting asset paths."""
    if tree.get("sha") != MENAGERIE_REVISION or tree.get("truncated") is not False:
        raise ValueError("Expected the complete release-pinned Menagerie tree")
    result, seen = [], set()
    for entry in tree["tree"]:
        path = PurePosixPath(entry["path"])
        if not path.parts or path.parts[0] != SUBTREE or entry["type"] == "tree":
            continue
        if (path.is_absolute() or ".." in path.parts or "\\" in str(path)
                or entry["type"] != "blob" or entry.get("mode") not in {"100644", "100755"}
                or len(path.parts) < 2 or str(path) in seen):
            raise ValueError(f"Unsafe or duplicate Panda asset: {path}")
        sha = entry.get("sha", "")
        if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(f"Invalid Git blob hash: {path}")
        if not isinstance(entry.get("size"), int) or not 0 <= entry["size"] <= 64 * 1024 * 1024:
            raise ValueError(f"Invalid asset size: {path}")
        seen.add(str(path))
        result.append(entry)
    if not {f"{SUBTREE}/panda.xml", f"{SUBTREE}/scene.xml"} <= seen:
        raise ValueError("Pinned inventory lacks Panda model and scene")
    result = sorted(result, key=lambda entry: entry["path"])
    canonical = [{key: entry[key] for key in ("path", "type", "mode", "sha", "size")} for entry in result]
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != PANDA_INVENTORY_SHA256:
        raise ValueError("Panda inventory does not match the complete pinned subtree")
    return result


def restore_panda(output_dir: Path, *, tree: dict | None = None) -> dict:
    """Atomically install absent/empty assets; existing models are never replaced."""
    output_dir = Path(output_dir)
    if output_dir.is_symlink() or (output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    )):
        raise ValueError("Model output directory must be absent or empty")
    if tree is None:
        tree = json.loads(_fetch(f"https://api.github.com/repos/{REPOSITORY}/git/trees/{MENAGERIE_REVISION}?recursive=1"))
    entries = panda_inventory(tree)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".panda-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / SUBTREE
        staged.mkdir()

        def download(entry: dict) -> tuple[str, dict]:
            data = _fetch(f"https://raw.githubusercontent.com/{REPOSITORY}/{MENAGERIE_REVISION}/{entry['path']}")
            blob_hash = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if len(data) != entry["size"] or blob_hash != entry["sha"]:
                raise ValueError(f"Pinned Git blob mismatch: {entry['path']}")
            relative = PurePosixPath(entry["path"]).relative_to(SUBTREE)
            path = staged / str(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return relative.as_posix(), {"git_blob_sha1": blob_hash, "sha256": hashlib.sha256(data).hexdigest(),
                                         "size_bytes": len(data)}

        with ThreadPoolExecutor(max_workers=4) as pool:
            files = dict(pool.map(download, entries))
        if output_dir.exists():
            output_dir.rmdir()  # Also rejects a concurrent writer populating it.
        staged.rename(output_dir)
    return {"schema_version": 1, "repository": REPOSITORY, "revision": MENAGERIE_REVISION,
            "inventory_sha256": PANDA_INVENTORY_SHA256,
            "subtree": SUBTREE, "files": files, "output_dir": str(output_dir.resolve()),
            "scope": "Git-blob-verified asset snapshot; not original producer environment attestation"}
