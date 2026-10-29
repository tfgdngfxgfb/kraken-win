"""Build hash-verified local SymPy/W&B compatibility wheels; preserve native payloads."""

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = {
    "sympy-1.14.0-py3-none-any.whl": (
        "a2/09/77d55d46fd61b4a135c444fc97158ef34a095e5681d0a6c10b75bf356191",
        "e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5",
    ),
    "sympy-1.14.0.tar.gz": (
        "83/d3/803453b36afefb7c2bb238361cd4ae6125a569b4db67cd9e79846ba2d68c",
        "d3d3fe8df1e5a0b42f0e7bdf50541697dbe7d23746e894990c030e2b05e72517",
    ),
    "wandb-0.31.0-py3-none-win_amd64.whl": (
        "90/7a/aaee2e55d1aa6f855d01e702263e2e0ef733da45583ecd9836d68ea3d67d",
        "488f8ef63cae217ba220e6d8c163ee8c5b98320ee1866ce774c4da0847235f02",
    ),
    "wandb-0.31.0-py3-none-manylinux_2_28_x86_64.whl": (
        "2a/07/5594faeaea9440be2ea01cee692573416b3997879df67f0e940bfbfc2d26",
        "009bcf370598d5e7a3d3bb66e33510f78c17fcc0505f8aa7ecf51dbc4378f979",
    ),
    "wandb-0.31.0.tar.gz": (
        "07/a5/8fc1c319f2db9be5a707c25a76ecba1b2b9450a44a05b295ff5aa379bbfa",
        "dcdaa635576ff17b18a757b39fc22b0db3d1c6e01d5ce090fe80f6adc776028d",
    ),
    "mpmath-1.4.1-py3-none-any.whl": (
        "13/f2/abeec3db71d221ccd3cd89d206be1fabf5a3ee7178862f5fba23a59607e0",
        "dc4f0ea2304480d4a9a48a94c1020571558ade522b44a6912efac63a586e140f",
    ),
    "cwevals-0.0.1b3-py3-none-any.whl": (
        "0c/d0/89df787a28354209df959267b24a56cd6ade8709098e146f286f1ac320e6",
        "3ff43ebab38a553877da8951305924a39a6f30809aae7f567de9ed1104d7ef15",
    ),
    "cwevals-0.0.1b3.tar.gz": (
        "d9/c3/f55a9932e883c768ff8abc1e9bf61d946caa30a5a5e2d580176614cd072a",
        "2647ffa8338634db70e9cbd1d9232f0a62f2b7d7b2c5d8a2f0bb750d4aeced45",
    ),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(name, directory):
    folder, digest = ARTIFACTS[name]
    path = directory / name
    if not path.exists():
        data = urllib.request.urlopen(
            f"https://files.pythonhosted.org/packages/{folder}/{name}", timeout=90
        ).read()
        if sha(data) != digest:
            raise RuntimeError(f"Upstream SHA256 mismatch: {name}")
        path.write_bytes(data)
    if sha(path.read_bytes()) != digest:
        raise RuntimeError(f"Cached SHA256 mismatch: {name}")
    return path


def replace_once(data, old, new):
    if data.count(old) != 1:
        raise RuntimeError(f"Unexpected upstream content for {old!r}")
    return data.replace(old, new)


def patch_wheel(
    original,
    destination,
    name,
    version,
    dependency_old,
    dependency_new,
    module,
    local_tag="pylaia1",
):
    local_version = version + "+" + local_tag
    old_info = f"{name}-{version}.dist-info/"
    new_info = f"{name}-{local_version}.dist-info/"
    with zipfile.ZipFile(original) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise RuntimeError("Invalid upstream wheel")
        # ZipInfo normalizes backslashes on Windows: inspect the raw names too.
        for path in names + [info.orig_filename for info in archive.infolist()]:
            if (
                PurePosixPath(path).is_absolute()
                or PureWindowsPath(path).drive
                or ".." in PurePosixPath(path).parts
                or "\\" in path
            ):
                raise RuntimeError("Unsafe wheel path")
        files = {path: archive.read(path) for path in names}
        attributes = {info.filename: info for info in archive.infolist()}
    # Verify the original RECORD before any modification.
    records = list(csv.reader(io.StringIO(files[old_info + "RECORD"].decode())))
    if (
        any(len(row) != 3 for row in records)
        or len({row[0] for row in records}) != len(records)
        or {row[0] for row in records}
        != {path for path in files if not path.endswith("/")}
    ):
        raise RuntimeError("Incomplete or duplicate upstream RECORD")
    for path, digest, size in records:
        if path == old_info + "RECORD":
            if digest or size:
                raise RuntimeError("Unexpected upstream RECORD self-hash")
        else:
            expected = (
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(files[path]).digest())
                .rstrip(b"=")
                .decode()
            )
            if digest != expected or int(size) != len(files[path]):
                raise RuntimeError(f"Invalid upstream RECORD: {path}")
    patches = {
        old_info + "METADATA": replace_once(
            replace_once(
                files[old_info + "METADATA"],
                f"Version: {version}\n".encode(),
                f"Version: {local_version}\n".encode(),
            ),
            dependency_old,
            dependency_new,
        ),
        module: replace_once(
            files[module],
            f'__version__ = "{version}"'.encode(),
            f'__version__ = "{local_version}"'.encode(),
        ),
    }
    output = {
        path.replace(old_info, new_info, 1)
        if path.startswith(old_info)
        else path: patches.get(path, data)
        for path, data in files.items()
        if path != old_info + "RECORD"
    }
    records = io.StringIO(newline="")
    writer = csv.writer(records, lineterminator="\n")
    for path, data in sorted(output.items()):
        if path.endswith("/"):
            continue
        writer.writerow(
            (
                path,
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .rstrip(b"=")
                .decode(),
                len(data),
            )
        )
    writer.writerow((new_info + "RECORD", "", ""))
    output[new_info + "RECORD"] = records.getvalue().encode()
    target = destination / original.name.replace(
        f"{name}-{version}-", f"{name}-{local_version}-", 1
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in sorted(output.items()):
            original_path = (
                path.replace(new_info, old_info, 1)
                if path.startswith(new_info)
                else path
            )
            info = copy.copy(attributes[original_path])
            info.filename = path
            info.date_time = (2026, 10, 20, 0, 0, 0)
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
    data = buffer.getvalue()
    if target.exists() and target.read_bytes() != data:
        raise RuntimeError(f"Refusing to overwrite different wheel: {target}")
    target.write_bytes(data)
    return {
        "wheel": target.name,
        "sha256": sha(data),
        "upstream": original.name,
        "changed": sorted(patches),
        "native_payloads_unchanged": True,
    }


def wheel_platform():
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("Compatibility wheels require x86-64")
    if sys.platform == "win32":
        return "win_amd64"
    if sys.platform == "linux":
        return "manylinux_2_28_x86_64"
    raise RuntimeError("Compatibility wheels support only Windows and Linux")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "dist/python-compatibility"
    )
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--sympy-only", action="store_true", help="Build only SymPy/mpmath for the optional documentation bootstrap.")
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sources = destination / "sources"
    sources.mkdir(exist_ok=True)
    platform_tag = wheel_platform()
    for name in (
        "sympy-1.14.0.tar.gz",
        "wandb-0.31.0.tar.gz",
        "cwevals-0.0.1b3.tar.gz",
    ):
        if not args.sympy_only or name.startswith("sympy-"):
            fetch(name, sources)
    built = []
    for name, version, dependency_old, dependency_new, module in (
        (
            "sympy",
            "1.14.0",
            b"Requires-Dist: mpmath<1.4,>=1.1.0",
            b"Requires-Dist: mpmath==1.4.1",
            "sympy/release.py",
        ),
        (
            "wandb",
            "0.31.0",
            b"Requires-Dist: cwevals==0.0.1b1",
            b"Requires-Dist: cwevals==0.0.1b3",
            "wandb/__init__.py",
        ),
    ):
        if args.sympy_only and name != "sympy":
            continue
        tag = "any" if name == "sympy" else platform_tag
        original = fetch(f"{name}-{version}-py3-none-{tag}.whl", sources)
        built.append(
            patch_wheel(
                original,
                destination,
                name,
                version,
                dependency_old,
                dependency_new,
                module,
                local_tag="pylaia2" if name == "wandb" else "pylaia1",
            )
        )
    dependencies = [
        fetch(name, destination)
        for name in (
            "mpmath-1.4.1-py3-none-any.whl",
            "cwevals-0.0.1b3-py3-none-any.whl",
        )
        if not args.sympy_only or name.startswith("mpmath-")
    ]
    (destination / "build-info.json").write_text(
        json.dumps(
            {
                "local_wheels": built,
                "recipe": "maintenance/python-compat/build_compatibility.py",
                "upstream_artifacts": {
                    name: digest for name, (_, digest) in ARTIFACTS.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.install:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                *(str(destination / row["wheel"]) for row in built),
                *(str(path) for path in dependencies),
            ],
            check=True,
        )
    print(json.dumps(built, indent=2))


if __name__ == "__main__":
    main()
