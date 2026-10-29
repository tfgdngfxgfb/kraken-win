Windows 11
==========

This fork targets standard 64-bit CPython 3.14 and 3.15 with official PyTorch
2.15 and TorchVision 0.29.2 wheels. Free-threaded Python is not covered.
Use a virtual environment. Run these commands in PowerShell from this fork's
checkout. Enable UTF-8 mode in every new terminal: HTRMoPo 0.6 reads its Unicode
metadata using the interpreter's default encoding.

.. code-block:: powershell

   $env:PYTHONUTF8 = '1'
   py -3.15 -m venv .venv315
   .\.venv315\Scripts\python.exe -m pip install --upgrade pip
   .\.venv315\Scripts\python.exe -m pip install 'torch==2.15.0' 'torchvision==0.29.2' --index-url https://download.pytorch.org/whl/cpu
   .\.venv315\Scripts\python.exe maintenance/python-compat/build_compatibility.py --sympy-only --output .local/python-compat-wheels

For Python 3.14, replace ``-3.15`` with ``-3.14`` and use a separate environment.

CPU and NVIDIA CUDA
-------------------

The 2026-10-28 update uses unmodified official PyTorch 2.15.0 and TorchVision
0.29.2 Windows wheels. PyTorch binding compilation and private PyLaia binding
payloads are no longer used. A Python exit callback calls the official
``torch._C._cudnn_clear_dropout_state`` function before DLL unload: CUDA LSTM
dropout otherwise still crashes at process exit on Windows Python 3.15.
The official binding files remain unmodified.
CPU inference is available with ``-d cpu``.

For an NVIDIA GPU, install both official CUDA 13.2 packages in place of the CPU
packages before installing Kraken:

.. code-block:: powershell

   .\.venv315\Scripts\python.exe -m pip install --upgrade 'torch==2.15.0' 'torchvision==0.29.2' --index-url https://download.pytorch.org/whl/cu132
   .\.venv315\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

CUDA 13.2 needs a compatible NVIDIA driver. The separate CUDA toolkit is not
required to use the official wheels. CUDA, cuDNN and dropout remain enabled.
Use matching CPU or CUDA variants of both packages. The runtime checks require
PyTorch 2.15 or newer on Windows Python 3.15 and reject the old 2.14.1 runtime
before models are loaded.

Python 3.15 native dependencies
-------------------------------

The previously built Windows Python 3.15 scikit-image 0.26.0, PyYAML 6.0.3 and
Core ML parser wheels can be reused with this update. A Kraken wheel does not
contain its dependencies. If these wheels are available, pass their directory
as another ``--find-links`` argument when installing Kraken below.

For a source build of scikit-image, install Visual Studio Build Tools with the
C++ workload and Windows SDK, plus LLVM/Clang. Clang is needed for its Pythran
C++ code; ``include-v2=false`` matches upstream's wheel layout. No scikit-image
source modifications are required.

.. code-block:: powershell

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
   .\.venv315\Scripts\python.exe -m pip install --find-links .local/python-compat-wheels -c maintenance/dependency-constraints.txt -e '.[test,pdf]'
   .\.venv315\Scripts\python.exe -m pip check
   .\.venv315\Scripts\kraken.exe --help
   .\.venv315\Scripts\ketos.exe --help

Dependency versions
-------------------

The 2026-10-29 update selects the latest compatible runtime, test,
documentation and build libraries with PyLaia 1.2.11. This includes
platformdirs 4.14.0, jsonargparse 4.53.0, setuptools 84.2.1, certifi 2026.10.23,
pytest 9.2.0, pandas 3.1.0 and python-dateutil 3.0.0. Full exact selections are
recorded in maintenance/dependency-constraints.txt.

SymPy 1.14.0+pylaia1 pairs with mpmath 1.4.1. Optional W&B 0.31.0+pylaia2
pairs with cwevals 0.0.1b3. Their compatibility recipes preserve numerical
code and native backends. Documentation now uses official Sphinx 9.2.0,
Docutils 0.23 and local MyST 5.1.0+kraken2. See
``maintenance/python-compat/README.rst`` and
``maintenance/docs-compat/README.rst`` for recipes and licenses.

Use ``-c maintenance/dependency-constraints.txt`` to reproduce these versions,
along with ``--find-links`` folders for the local compatibility wheels.
Existing transfer ZIPs retain their recorded dependency sets and are not
changed by updating this source checkout. In particular, fork4 and earlier
ZIP installers still target the historical PyTorch 2.14.1 runtime.

PyLaia compatibility
--------------------

Kraken does not depend on the PyLaia Python package. The supplied PyLaia
``1.2.11+fork.20261028`` bundle uses the same official PyTorch 2.15.0 and
TorchVision 0.29.2 CPU/CUDA 13.2 wheels. It can share Kraken's Python 3.15
environment. The earlier 1.2.8 bundle still requires the older PyTorch runtime.

Extract the complete 1.2.11 ZIP and install its wheel into the Kraken environment
after the official CPU or CUDA runtime has been installed above. Replace the
bundle directory below with its actual location:

.. code-block:: powershell

   $pylaiaBundle = 'C:\path\PyLaia-1.2.11-fork.20261028-Windows-x64'
   .\.venv315\Scripts\python.exe -m pip install --find-links "$pylaiaBundle\wheels\cp315" --find-links "$pylaiaBundle\compatibility" -c maintenance/dependency-constraints.txt "$pylaiaBundle\wheels\pylaia-1.2.11+fork.20261028-py3-none-any.whl[language-model]"
   .\.venv315\Scripts\python.exe -m pip check

The language-model extra installs the supplied Flashlight/KenLM native wheels.
For optional W&B support, use ``[language-model,wandb]`` instead. Its local W&B
and cwevals wheels are in the bundle's compatibility directory. PyLaia and
Kraken both retain Python cleanup of cuDNN dropout state before shutdown;
the official Torch binding files are not modified.

The local integration checks cover both import orders, three CTC training
steps with finite gradients and updated weights, checkpoint round-trips,
Kraken OCR and segmentation, native language-model decoding and offline W&B.
Results are recorded in ``maintenance/dependency-update-20261029.json``.

The optional ``maintenance/windows315/verify_shared_pylaia.py`` helper can be
used to repeat the training checks. Run both import orders as separate
processes; omit ``--require-cuda`` on CPU:

.. code-block:: powershell

   .\.venv315\Scripts\python.exe -I maintenance/windows315/verify_shared_pylaia.py --import-order kraken-first --require-cuda
   .\.venv315\Scripts\python.exe -I maintenance/windows315/verify_shared_pylaia.py --import-order laia-first --require-cuda

The binding builder and assembler in that directory are historical tools for
reproducing old builds; they are not part of the new installation path.

Model formats and PDFs
----------------------

Both legacy ``.mlmodel`` and ``.safetensors`` models use PyTorch for inference.
Core ML's macOS-only native prediction runtime is not required. Its warnings
about missing native runtime extensions on Windows do not prevent loading
Kraken models. The standard line segmentation model is unchanged.

Install this fork's PDF extra to include PDFium and its prebuilt Windows library:

.. code-block:: powershell

   .\.venv315\Scripts\python.exe -m pip install --find-links .local/python-compat-wheels -c maintenance/dependency-constraints.txt -e '.[pdf]'

PDF pages are rendered at 300 dpi. No separate libvips installation is required.
Image input works without this optional dependency.

Local tests
-----------

.. code-block:: powershell

   $env:PYTHONUTF8 = '1'
   $env:OMP_NUM_THREADS = '1'
   $env:MKL_NUM_THREADS = '1'
   .\.venv315\Scripts\python.exe -m pytest -m 'not network and not slow'

The runtime regression checks exercise Python autograd callbacks and normal
process exit both with a direct call to the official cleanup API and after
importing Kraken, which registers that cleanup automatically. CUDA checks
also cover cuDNN LSTM dropout and backward propagation. GPU tests run when a
CUDA-enabled PyTorch installation and a GPU are available. Distributed
execution, ``torch.compile`` and free-threaded Python are outside this validation.

Use ``--workers 0`` when training from notebooks or scripts without a guarded
entry point. Command-line training supports worker processes. In Python
scripts using workers, place execution inside ``if __name__ == '__main__':``
so Windows can spawn workers safely.

Automatic Windows/Linux CI triggers remain paused. No CI run is started for
the 2026-10-28 update. The manual workflow now installs official PyTorch wheels
instead of building custom bindings; re-enabling CI requires a separate action.
