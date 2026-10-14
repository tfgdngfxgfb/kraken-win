The two Python build helpers are adapted from the user-supplied PyLaia
`1.2.3+fork.20261014` Windows bundle under its MIT license. The source file hashes,
bundle hash and source commit are recorded in `provenance.json`.

The official Windows PyTorch 2.14.1 cp315 wheels have an incompatible CPython
thread-state layout in Python autograd callbacks. The builder recompiles only the
three Python binding files against the running standard CPython 3.15 headers. It
uses the pinned upstream PyTorch source commit and SHA-256-verified official
computation libraries. CPU/CUDA kernels are not changed. The assembler updates
wheel version metadata and every RECORD hash. No installed environment is patched.

For a CPU build, run in a Visual Studio x64 developer PowerShell with Python 3.15:

```powershell
python maintenance/windows315/build_torch315.py --build CPU --work-dir .local/torch315-build --output-dir .local/torch315-wheels --jobs 2
python -m pip install --no-deps .local/torch315-wheels/torch-2.14.1+pylaia315cpu-cp315-cp315-win_amd64.whl
python -m pip install --no-deps torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

The Windows 3.15 CI job follows this path on `windows-2022` with Visual Studio
2022, matching the toolchain used for the supplied validated bindings. The newer
VS 2026 runner reproduces the SLEEF compiler-probe failure reported in
https://github.com/microsoft/vcpkg/issues/51763. Other matrix jobs use the latest
runner images. It needs MSVC x64 Build Tools, Windows
SDK, Git and network access. Existing build outputs are preserved. Full native
build logs remain local because compiler diagnostics can contain environment data.

For local use, the supplied PyLaia installer already provides corrected CPU and
CUDA 13.2 bindings, avoiding recompilation. Install Kraken into that environment
after installing its scikit-image dependency as described in the Windows guide.
CUDA binding rebuilds additionally require a matching CUDA 13.2/cuDNN 9.24 SDK;
the supplied PyLaia build guide contains that separate SDK preparation procedure.
Windows Python 3.15 validation covers one CPU device or one NVIDIA GPU. Distributed
training, free-threaded CPython and `torch.compile` are outside this validation.
