Windows 11
==========

This fork supports 64-bit CPython 3.14 only. Use the latest 3.14 maintenance release and a virtual environment. Run the following commands in PowerShell from this fork's checkout. Enable UTF-8 mode in every new terminal: HTRMoPo 0.6 reads its Unicode metadata using the interpreter's default encoding.

.. code-block:: powershell

   $env:PYTHONUTF8 = '1'
   py -3.14 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -e '.[test]'
   .\.venv\Scripts\kraken.exe --help
   .\.venv\Scripts\ketos.exe --help

CPU and NVIDIA CUDA
-------------------

CPU inference is available with ``-d cpu``. For NVIDIA GPUs, install a matching torch/torchvision pair from the official PyTorch index. The CUDA 13.0 build requires a compatible NVIDIA driver; the separate CUDA toolkit is not needed for these wheels.

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
