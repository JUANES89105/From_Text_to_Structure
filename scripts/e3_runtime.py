"""Expose only the existing E3 dependency closure; never install/download packages.

The shared .venv includes optional notebook/API packages whose macOS dataless
metadata can block Transformers availability probes. Relative symlinks isolate
E3 imports without deleting, hiding, or modifying any original installed package.
"""
import importlib.metadata
import os
from pathlib import Path
import site
import sys


def activate():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/e3_scalability"
    requirements = output / "environment_repair_requirements.txt"
    overlay = output / "runtime_site"
    overlay.mkdir(parents=True, exist_ok=True)
    targets = {}
    for line in requirements.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==")
        distribution = importlib.metadata.distribution(name)
        if distribution.version != version:
            raise RuntimeError(f"E3 runtime requires installed {name}=={version}; found {distribution.version}")
        for item in distribution.files or []:
            top = item.parts[0]
            if top in ("..", "."):
                continue
            target = Path(distribution.locate_file(top)).absolute()
            if target.exists():
                targets[top] = target
    for name, target in sorted(targets.items()):
        link = overlay / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"Unexpected E3 runtime link: {link}")
        elif link.exists():
            raise RuntimeError(f"Unexpected non-symlink in E3 runtime: {link}")
        else:
            link.symlink_to(os.path.relpath(target, overlay), target_is_directory=target.is_dir())
    original_sites = {str(Path(p).resolve()) for p in site.getsitepackages()}
    sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) not in original_sites]
    sys.path.insert(0, str(overlay))
    importlib.invalidate_caches()
    return {"strategy": "relative symlinks exposing only the pinned E3 dependency closure",
            "directory": str(overlay.relative_to(root)), "linked_top_level_entries": len(targets)}
