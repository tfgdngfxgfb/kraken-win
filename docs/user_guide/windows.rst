Windows 11
==========

This fork targets standard 64-bit CPython 3.14 and 3.15. Python 3.15 requires the corrected PyTorch bindings described below; free-threaded Python is not covered. Use a virtual environment. Run the following commands in PowerShell from this fork's checkout. Enable UTF-8 mode in every new terminal: HTRMoPo 0.6 reads its Unicode metadata using the interpreter's default encoding.

.. code-block:: powershell

   $env:PYTHONUTF8 = '1'
   py -3.14 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -e '.[test]'
   .\.venv\Scripts\kraken.exe --help
   .\.venv\Scripts\ketos.exe --help

Python 3.15 CPU and CUDA setup
-----------------------------

The official PyTorch 2.14.1 Windows cp315 wheels have an incompatible Python thread-state ABI. Python autograd callbacks fail on CPU, and CUDA can crash at process shutdown. Use the corrected CPU or CUDA 13.2 bindings supplied with the private PyLaia ``1.2.3+fork.20261014`` bundle. Kraken does not require the PyLaia Python package itself; both applications can share the corrected PyTorch runtime. Ordinary PyPI dependency resolution alone does not install these corrected bindings.

For a shared environment, extract the complete PyLaia ZIP and run its installer with an explicit environment path inside the Kraken checkout. Replace the bundle path below with your extracted folder. Choose ``CPU`` instead of ``CUDA`` if needed. The environment path must not already exist.

.. code-block:: powershell

   & 'C:\path\PyLaia-1.2.3-fork.20261014-Windows-x64\Install-PyLaia.ps1' -PythonVersion 3.15 -TorchBuild CUDA -InstallationPath "$PWD\.venv315"

For Kraken alone, the source builders in ``maintenance/windows315`` produce the same corrected CPU bindings without installing PyLaia. Run these commands in an x64 Visual Studio developer PowerShell, then continue with scikit-image below:

.. code-block:: powershell

   py -3.15 -m venv .venv315
   .\.venv315\Scripts\python.exe maintenance/windows315/build_torch315.py --build CPU --work-dir .local/torch315-build --output-dir .local/torch315-wheels --jobs 2
   .\.venv315\Scripts\python.exe -m pip install --no-deps .local/torch315-wheels/torch-2.14.1+pylaia315cpu-cp315-cp315-win_amd64.whl
   .\.venv315\Scripts\python.exe -m pip install --no-deps torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu

As of 2026-10-14, ``scikit-image 0.26.0`` and ``PyYAML 6.0.3`` need local source builds on Windows Python 3.15. Install Visual Studio Build Tools with the C++ workload and Windows SDK, plus LLVM/Clang. The commands below assume ``clang-cl.exe`` is on ``PATH``. A portable LLVM installation can also be added to ``PATH``. Clang is needed for scikit-image's Pythran C++ code; the default MSVC build fails. No changes to scikit-image's source are required. ``include-v2=false`` matches upstream's wheel layout.

.. code-block:: powershell

   $env:PYTHONUTF8 = '1'
   .\.venv315\Scripts\python.exe -m pip install --upgrade pip
   # Add the bin directory of your LLVM installation if necessary.
   $env:PATH = 'C:\Program Files\LLVM\bin;' + $env:PATH
   $previousCC = $env:CC
   $previousCXX = $env:CXX
   try {
       $env:CC = 'clang-cl'
       $env:CXX = 'clang-cl'
       .\.venv315\Scripts\python.exe -m pip install 'scikit-image>=0.26.0,<0.27' --config-settings 'setup-args=--vsenv' --config-settings 'setup-args=-Dinclude-v2=false'
       if ($LASTEXITCODE -ne 0) { throw 'scikit-image build failed' }
   } finally {
       $env:CC = $previousCC
       $env:CXX = $previousCXX
   }
   .\.venv315\Scripts\python.exe -m pip install -e '.[test,pdf]'
   .\.venv315\Scripts\python.exe -m pip check

The PyLaia bundle already includes a PyYAML wheel. A locally built scikit-image wheel can be installed instead of compiling it again. Keep these native wheels and the corrected PyTorch wheel when transferring an installation to a machine without build tools. A Kraken wheel alone does not contain its dependencies. The existing Python 3.14 transfer package is unchanged.

Python 3.15 CUDA 13.2 needs a compatible NVIDIA driver. The separate CUDA toolkit is not required to use the supplied wheels. Do not replace the corrected PyTorch wheel with an official cp315 wheel. Kraken checks the binding marker before loading models and releases cuDNN's dropout cache before process shutdown; cuDNN and dropout remain enabled. Validation covers a single CPU device or NVIDIA GPU, including normal exit. Distributed execution and ``torch.compile`` are outside this validation.

CPU and NVIDIA CUDA
-------------------

CPU inference is available with ``-d cpu``. On Python 3.14, install a matching torch/torchvision pair from the official PyTorch index for NVIDIA GPUs. The CUDA 13.0 build requires a compatible NVIDIA driver; the separate CUDA toolkit is not needed for these wheels. Use the corrected CUDA 13.2 setup above on Python 3.15.

.. code-block:: powershell

   .\.venv\Scripts\python.exe -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130
   .\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   .\.venv\Scripts\kraken.exe -d cuda:0 -i image.png lines.json segment -bl

Use ``--workers 0`` when training from notebooks or scripts without a guarded entry point. Command-line training supports worker processes. For Python scripts using workers, place execution inside ``if __name__ == '__main__':`` so Windows can spawn workers safely.

Model formats and PDFs
----------------------

Both legacy ``.mlmodel`` and ``.safetensors`` models use PyTorch for inference. Core ML's macOS-only native prediction runtime is not required. Its warnings about missing native runtime extensions on Windows do not prevent loading kraken models.

Install this fork's PDF extra to include PDFium and its prebuilt Windows library:

.. code-block:: powershell

   .\.venv\Scripts\python.exe -m pip install -e '.[pdf]'

PDF pages are rendered at 300 dpi. No separate libvips installation is required. Image input works without this optional dependency.

Tests
-----

.. code-block:: powershell

   $env:PYTHONUTF8 = '1'
   $env:OMP_NUM_THREADS = '1'
   $env:MKL_NUM_THREADS = '1'
   .\.venv\Scripts\python.exe -m pytest -m 'not network and not slow'

CUDA regression tests run when a CUDA-enabled torch installation and GPU are available. Windows and Linux CI cover the CPU path.
