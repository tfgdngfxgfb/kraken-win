# Adapted from PyLaia 1.2.3+fork.20261014; see LICENSE-PyLaia.txt.
"""Assemble a normal Torch wheel from verified official cores and local bindings.

Only the CPython-facing bindings and their import libraries are replaced. The
official CPU/CUDA kernels, Python modules, package metadata and licenses remain
in the wheel. No installed environment is patched in place.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

SOURCE_COMMIT = "5c4886908584029761b579af026dcfb627c84070"
BUILDS = {
    "CPU": {
        "base": "2.14.1+cpu",
        "version": "2.14.1+pylaia315cpu",
        "index": "cpu",
        "sha256": "e799161f94d97485857ec82424e40a83d7648ab02c03e3d6e71c7838a4d7b89e",
    },
    "CUDA": {
        "base": "2.14.1+cu132",
        "version": "2.14.1+pylaia315cu132",
        "index": "cu132",
        "sha256": "f9a7336a67a7a4a29cb568adecc8aef2b9cbb6bbd1d0efc2a52b9e8f7b5fb66b",
    },
}
BINDING_FILES = {
    "torch/_C.cp315-win_amd64.pyd",
    "torch/lib/torch_python.dll",
    "torch/lib/torch_python.lib",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_payload(directory, build):
    manifest = json.loads((directory / "build-info.json").read_text(encoding="utf-8"))
    if (
        manifest["source_commit"] != SOURCE_COMMIT
        or manifest["version"] != build["version"]
        or manifest["base_version"] != build["base"]
        or set(manifest["files"]) != BINDING_FILES
    ):
        raise ValueError("Unexpected Python 3.15 binding payload")
    files = {}
    for wheel_path, entry in manifest["files"].items():
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"Unsafe payload path: {wheel_path}")
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Binding checksum mismatch: {wheel_path}")
        files[wheel_path] = path
    return files


def download_original(directory, build):
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"torch-{build['base']}-cp315-cp315-win_amd64.whl"
    target = directory / filename
    if target.exists():
        return target
    url = f"https://download.pytorch.org/whl/{build['index']}/" + filename.replace(
        "+", "%2B"
    )
    with tempfile.NamedTemporaryFile(
        dir=directory, suffix=".download", delete=False
    ) as out:
        temporary = Path(out.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as source:
                shutil.copyfileobj(source, out)
        except BaseException:
            out.close()
            temporary.unlink(missing_ok=True)
            raise
    if digest(temporary) != build["sha256"]:
        temporary.unlink()
        raise ValueError("Official PyTorch download checksum mismatch")
    temporary.rename(target)
    return target


def assemble(original, payload, output_directory, build):
    bindings = load_payload(payload, build)
    if digest(original) != build["sha256"]:
        raise ValueError("Original PyTorch wheel checksum mismatch")
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / (
        f"torch-{build['version']}-cp315-cp315-win_amd64.whl"
    )
    if destination.exists():
        raise FileExistsError(f"Preserving existing wheel: {destination}")
    old_info = f"torch-{build['base']}.dist-info"
    new_info = f"torch-{build['version']}.dist-info"
    records = []
    with tempfile.TemporaryDirectory(dir=output_directory, prefix="torch315-") as tmp:
        temporary = Path(tmp) / destination.name
        with (
            zipfile.ZipFile(original) as source,
            zipfile.ZipFile(
                temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
            ) as target,
        ):
            names = source.namelist()
            if len(names) != len(set(names)) or not BINDING_FILES.issubset(names):
                raise ValueError("Unexpected original Torch wheel contents")
            for name in names:
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "\\" in name:
                    raise ValueError(f"Unsafe original wheel entry: {name}")
                if name.endswith("/") or name == old_info + "/RECORD":
                    continue
                new_name = name.replace(old_info + "/", new_info + "/", 1)
                info = zipfile.ZipInfo(new_name, date_time=(2026, 10, 14, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = source.getinfo(name).external_attr
                checksum = hashlib.sha256()
                length = 0
                if name in bindings:
                    input_stream = bindings[name].open("rb")
                elif name in ("torch/version.py", old_info + "/METADATA"):
                    data = source.read(name).decode("utf-8")
                    if name == "torch/version.py":
                        data, count = re.subn(
                            r"^__version__ = .+$",
                            f"__version__ = {build['version']!r}",
                            data,
                            flags=re.MULTILINE,
                        )
                        data += "\n_pylaia_python315_abi = 1\n"
                    else:
                        data, count = re.subn(
                            r"^Version: .+$",
                            f"Version: {build['version']}",
                            data,
                            flags=re.MULTILINE,
                        )
                    if count != 1:
                        raise ValueError(f"Unexpected version metadata: {name}")
                    input_stream = io.BytesIO(data.encode("utf-8"))
                else:
                    input_stream = source.open(name)
                with input_stream, target.open(info, "w", force_zip64=True) as out:
                    while chunk := input_stream.read(1024 * 1024):
                        out.write(chunk)
                        checksum.update(chunk)
                        length += len(chunk)
                encoded = (
                    base64.urlsafe_b64encode(checksum.digest()).rstrip(b"=").decode()
                )
                records.append((new_name, "sha256=" + encoded, length))
            record_name = new_info + "/RECORD"
            records.append((record_name, "", ""))
            record = io.StringIO(newline="")
            csv.writer(record).writerows(records)
            target.writestr(record_name, record.getvalue())
        temporary.rename(destination)
    print(f"Assembled verified PyTorch wheel: {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", choices=BUILDS, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--original-wheel", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build = BUILDS[args.build]
    original = args.original_wheel or download_original(
        args.output_dir / "original", build
    )
    assemble(original, args.payload, args.output_dir, build)


if __name__ == "__main__":
    main()
