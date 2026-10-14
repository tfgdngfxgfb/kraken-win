# Adapted from PyLaia 1.2.3+fork.20261014; see LICENSE-PyLaia.txt.
"""Rebuild Windows Torch 2.14.1 Python bindings against the actual 3.15 headers.

Run from an x64 Visual Studio developer PowerShell. Computation libraries use a
SHA-256-verified official wheel, not rebuilt or modified. Build outputs are
isolated from both the system Python and the target Kraken/PyLaia environment.
"""

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import venv
import zipfile
from pathlib import Path

from assemble_torch315_wheel import (
    BUILDS,
    SOURCE_COMMIT,
    assemble,
    digest,
    download_original,
)
def replace_source(path, before, after):
    content = path.read_text(encoding="utf-8")
    if content.count(before) != 1:
        raise ValueError(f"Unexpected source content: {path}")
    path.write_text(content.replace(before, after), encoding="utf-8", newline="\n")

ROOT = Path(__file__).resolve().parents[2]
UV_COMMIT = "840404ce8ba7cc0204be52389a6cfff9f2c90fb6"
SUBMODULES = (
    "pybind11",
    "protobuf",
    "onnx",
    "gloo",
    "fmt",
    "flatbuffers",
    "nlohmann",
    "cpp-httplib",
    "kineto",
    "NVTX",
    "cpuinfo",
    "mimalloc",
    "FP16",
    "FXdiv",
    "pthreadpool",
    "pocketfft",
    "ittapi",
    "psimd",
    "sleef",
)


def run(*args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def report_native_failure(log):
    """Show bounded compiler/CMake diagnostics without dumping the environment."""
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = set()
    pattern = re.compile(r"CMake Error|fatal error|error (?:C|LNK)\d+|Could NOT find|ninja: build stopped|^\s*FAILED:")
    for index, line in enumerate(lines):
        if pattern.search(line):
            selected.add(index)
            if "CMake Error" in line or "Could NOT find" in line:
                selected.update(range(index + 1, min(index + 7, len(lines))))
    print("Native build diagnostics (full log remains local):", flush=True)
    for index in sorted(selected)[:60]:
        line = lines[index]
        if re.search(r"(?i)token|password|secret|authorization|credential", line):
            print("[sensitive diagnostic omitted]", flush=True)
        else:
            print(line[:500], flush=True)


def short_path(path):
    result = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(path), result, len(result)):
        raise OSError(f"Cannot resolve compiler-safe path: {path}")
    return Path(result.value)


def clone(path, repository, tag, commit):
    if not path.exists():
        run(
            "git",
            "-c",
            "core.longpaths=true",
            "clone",
            "--depth",
            "1",
            "--branch",
            tag,
            repository,
            path,
        )
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise ValueError(f"Unexpected source commit: {path}")
    run("git", "-C", path, "config", "core.longpaths", "true")


def patch_bindings(source):
    marker = source / ".pylaia315-cmake-patched"
    if marker.exists():
        return
    replace_source(
        source / "cmake/PreBuildSteps.cmake",
        "if(NOT DEFINED USE_SYSTEM_LIBS OR NOT USE_SYSTEM_LIBS)",
        "if(NOT BUILD_PYTHON_ONLY AND (NOT DEFINED USE_SYSTEM_LIBS OR NOT USE_SYSTEM_LIBS))",
    )
    for file, target, option in (
        ("c10/CMakeLists.txt", "c10", "C10_BUILD_SHARED_LIBS"),
        ("c10/cuda/CMakeLists.txt", "c10_cuda", "C10_CUDA_BUILD_SHARED_LIBS"),
    ):
        replace_source(
            source / file,
            f"if(MSVC AND {option})",
            f"if(MSVC AND {option} AND TARGET {target})",
        )
    cmake = source / "torch/CMakeLists.txt"
    replace_source(
        cmake,
        "add_subdirectory(${LIBSHM_SRCDIR})",
        """if(MSVC AND BUILD_LIBTORCHLESS)
  add_library(shm SHARED IMPORTED)
  set_target_properties(shm PROPERTIES
    IMPORTED_IMPLIB "$ENV{LIBTORCH_LIB_PATH}/shm.lib"
    IMPORTED_LOCATION "$ENV{LIBTORCH_LIB_PATH}/shm.dll")
else()
  add_subdirectory(${LIBSHM_SRCDIR})
endif()""",
    )
    replace_source(
        cmake,
        "add_library(torch_python SHARED ${TORCH_PYTHON_SRCS})",
        """add_library(torch_python SHARED ${TORCH_PYTHON_SRCS})
if(MSVC AND BUILD_LIBTORCHLESS)
  set_source_files_properties(csrc/utils/device_lazy_init.cpp PROPERTIES COMPILE_DEFINITIONS WIN32)
endif()""",
    )
    link = "target_link_libraries(torch_python PRIVATE ${TORCH_LIB} ${TORCH_PYTHON_LINK_LIBRARIES})"
    replace_source(
        cmake,
        link,
        link
        + """
if(MSVC AND BUILD_LIBTORCHLESS)
  target_link_libraries(torch_python PRIVATE
    ${TORCH_CPU_LIB} "$ENV{LIBTORCH_LIB_PATH}/c10.lib")
  if(USE_CUDA)
    target_link_libraries(torch_python PRIVATE
      ${TORCH_CUDA_LIB} "$ENV{LIBTORCH_LIB_PATH}/c10_cuda.lib"
      "$ENV{CUDA_PATH}/lib/x64/cudart.lib")
  endif()
endif()
""",
    )
    replace_source(
        cmake,
        'install(TARGETS torch_python DESTINATION "${TORCH_INSTALL_LIB_DIR}")',
        'install(TARGETS torch_python DESTINATION "${TORCH_INSTALL_LIB_DIR}" COMPONENT pylaia315)',
    )
    replace_source(
        cmake,
        '    RUNTIME DESTINATION "."\n    LIBRARY DESTINATION "."\n    ARCHIVE DESTINATION "lib"',
        '    RUNTIME DESTINATION "." COMPONENT pylaia315\n'
        '    LIBRARY DESTINATION "." COMPONENT pylaia315\n'
        '    ARCHIVE DESTINATION "lib" COMPONENT pylaia315',
    )
    # Unused source-only submodules are not downloaded for the binding build.
    replace_source(source / "pyproject.toml", '    "third_party/**/LICENSE.rst",\n', "")
    marker.write_text(
        "CMake-only binding build adjustments; no C++ algorithm changes.\n"
    )


def export_payload(build, payload, spec, source, uv_source):
    artifacts = (
        ("torch/_C.cp315-win_amd64.pyd", build / "torch/_C.cp315-win_amd64.pyd"),
        ("torch/lib/torch_python.dll", build / "bin/torch_python.dll"),
        ("torch/lib/torch_python.lib", build / "lib/torch_python.lib"),
    )
    for _, path in artifacts:
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Missing completed binding artifact: {path}")
    payload.mkdir(parents=True)
    files = {}
    for wheel_path, built_path in artifacts:
        target = payload / built_path.name
        shutil.copy2(built_path, target)
        files[wheel_path] = {"path": target.name, "sha256": digest(target)}
    shutil.copy2(source / "LICENSE", payload / "LICENSE-PyTorch")
    shutil.copy2(uv_source / "LICENSE", payload / "LICENSE-libuv")
    manifest = {
        "version": spec["version"],
        "base_version": spec["base"],
        "source_commit": SOURCE_COMMIT,
        "official_wheel_sha256": spec["sha256"],
        "python_version": sys.version,
        "libuv_build_commit": UV_COMMIT,
        "files": files,
    }
    (payload / "build-info.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", choices=BUILDS, default="CPU")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist/pytorch315")
    parser.add_argument("--original-wheel", type=Path)
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if (
        sys.platform != "win32"
        or sys.version_info[:2] != (3, 15)
        or sys.maxsize <= 2**32
        or sysconfig.get_config_var("Py_GIL_DISABLED")
    ):
        parser.error("This builder requires Windows x64 Python 3.15")
    if args.build == "CUDA" and not args.cuda_root:
        parser.error("CUDA bindings require a local CUDA 13.2/cuDNN 9.24 SDK")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    work = (
        args.work_dir or ROOT / f"tmp/torch315-rebuild-{args.build.lower()}"
    ).resolve()
    work.mkdir(parents=True, exist_ok=True)
    work = short_path(work)
    output = args.output_dir.resolve()
    payload = output / args.build.lower()
    if payload.exists():
        raise FileExistsError(f"Preserving existing binding payload: {payload}")
    spec = BUILDS[args.build]
    original = args.original_wheel or download_original(work / "original", spec)
    if digest(original) != spec["sha256"]:
        raise ValueError("Official Torch source wheel checksum mismatch")
    core = work / "official-wheel"
    marker = core / ".verified-wheel-sha256"
    if not marker.exists():
        if core.exists():
            raise FileExistsError(f"Preserving incomplete extraction: {core}")
        with zipfile.ZipFile(original) as archive:
            archive.extractall(core)
        marker.write_text(spec["sha256"])
    if marker.read_text() != spec["sha256"]:
        raise ValueError("Unexpected extracted Torch core")
    source = work / "pytorch-source"
    clone(source, "https://github.com/pytorch/pytorch.git", "v2.14.1", SOURCE_COMMIT)
    submodules = list(SUBMODULES)
    if args.build == "CUDA":
        submodules += ["cudnn_frontend", "cutlass"]
    run(
        "git",
        "-C",
        source,
        "submodule",
        "update",
        "--init",
        "--depth",
        "1",
        "--",
        *(f"third_party/{name}" for name in submodules),
    )
    patch_bindings(source)
    environment = work / "build-env"
    if not environment.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "Scripts/python.exe"
    run(
        python,
        "-m",
        "pip",
        "install",
        "setuptools==84.0.0",
        "scikit-build-core==1.1.1",
        "numpy==2.5.4",
        "packaging==26.3",
        "six==1.17.0",
        "typing-extensions==4.16.0",
        "cmake==4.4.4",
        "ninja==1.13.2",
        "Cython==3.3.0",
        "PyYAML==6.0.3",
    )
    ninja_binary = Path(subprocess.check_output(
        [str(python), "-c", "import ninja; from pathlib import Path; print(Path(ninja.BIN_DIR) / 'ninja.exe')"],
        text=True,
    ).strip())
    if not ninja_binary.is_file():
        raise FileNotFoundError(f"Missing native Ninja executable: {ninja_binary}")
    uv_source = work / "libuv-source"
    clone(uv_source, "https://github.com/libuv/libuv.git", "v1.53.0", UV_COMMIT)
    uv_install = work / "libuv-install"
    if not (uv_install / "bin/uv.dll").exists():
        run(
            python,
            "-m",
            "cmake",
            "-S",
            uv_source,
            "-B",
            work / "libuv-build",
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_MAKE_PROGRAM={ninja_binary}",
            "-DLIBUV_BUILD_SHARED=ON",
            "-DBUILD_TESTING=OFF",
            f"-DCMAKE_INSTALL_PREFIX={uv_install}",
        )
        run(
            python,
            "-m",
            "cmake",
            "--build",
            work / "libuv-build",
            "--parallel",
            args.jobs,
        )
        run(python, "-m", "cmake", "--install", work / "libuv-build")
    build = work / "bindings-build"
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs),
        LIBTORCH_LIB_PATH=str(core / "torch/lib"),
        libuv_ROOT=str(uv_install),
        PYTORCH_BUILD_VERSION=spec["version"],
        PYTORCH_BUILD_NUMBER="1",
    )
    env["LIB"] = str(core / "torch/lib") + ";" + env.get("LIB", "")
    env["PATH"] = str(environment / "Scripts") + ";" + env["PATH"]
    definitions = dict.fromkeys(
        (
            "BUILD_PYTHON_ONLY",
            "BUILD_PYTHON",
            "BUILD_FUNCTORCH",
            "BUILD_LAZY_TS_BACKEND",
            "USE_ITT",
            "USE_DISTRIBUTED",
            "USE_GLOO",
            "USE_KINETO",
            "USE_NUMPY",
        ),
        "ON",
    )
    definitions.update(
        dict.fromkeys(
            (
                "BUILD_TEST",
                "BUILD_BINARY",
                "USE_XPU",
                "USE_ROCM",
                "USE_MKLDNN",
                "USE_NNPACK",
                "USE_PYTORCH_QNNPACK",
                "USE_XNNPACK",
                "USE_FBGEMM",
                "USE_MSLK",
                "USE_FLASH_ATTENTION",
                "USE_MEM_EFF_ATTENTION",
                "USE_MPI",
                "USE_TENSORPIPE",
                "USE_NCCL",
                "USE_SYSTEM_EIGEN_INSTALL",
                "USE_CUSPARSELT",
            ),
            "OFF",
        )
    )
    cuda_enabled = "ON" if args.build == "CUDA" else "OFF"
    definitions.update(
        dict.fromkeys(("USE_CUDA", "USE_CUDNN", "USE_NVRTC"), cuda_enabled)
    )
    base = short_path(Path(sys.base_prefix))
    definitions.update(
        {
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_MAKE_PROGRAM": str(ninja_binary),
            "CMAKE_C_COMPILER": "cl",
            "CMAKE_CXX_COMPILER": "cl",
            "BLAS": "Eigen",
            "Python_EXECUTABLE": str(python),
            "Python_INCLUDE_DIR": str(base / "include"),
            "Python_LIBRARY": str(base / "libs/python315.lib"),
            "CMAKE_LIBRARY_PATH": str(core / "torch/lib"),
            "CMAKE_CXX_FLAGS": f'/utf-8 /bigobj /FS /I"{core / "torch/include"}"',
            "CMAKE_C_FLAGS": "/utf-8",
        }
    )
    if args.cuda_root:
        sdk = short_path(args.cuda_root.resolve())
        env["CUDA_PATH"] = env["CUDNN_ROOT"] = str(sdk)
        env["PATH"] = str(sdk / "bin") + ";" + env["PATH"]
        env["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;9.0;10.0;12.0"
        definitions.update(
            {
                "CUDAToolkit_ROOT": str(sdk),
                "CUDNN_INCLUDE_DIR": str(sdk / "include"),
                "CUDNN_LIBRARY": str(sdk / "lib/x64/cudnn.lib"),
                "TORCH_CUDA_ARCH_LIST": env["TORCH_CUDA_ARCH_LIST"],
            }
        )
    command = [
        python,
        "-m",
        "pip",
        "install",
        "-e",
        source,
        "-v",
        "--no-build-isolation",
        "--no-deps",
        "--no-index",
        f"--config-settings=build-dir={build}",
        "--config-settings=build.targets=torch_python;_C",
        "--config-settings=install.components=pylaia315",
    ]
    command += [
        f"--config-settings=cmake.define.{key}={value}"
        for key, value in definitions.items()
    ]
    log = work / "bindings-build.log"
    # NVCC failures can dump environment variables: never publish the full log.
    with log.open("w", encoding="utf-8") as stream:
        try:
            run(*command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            report_native_failure(log)
            raise RuntimeError(
                f"Native binding build failed; inspect the local log: {log}"
            ) from None
    export_payload(build, payload, spec, source, uv_source)
    assemble(original, payload, output, spec)


if __name__ == "__main__":
    main()
