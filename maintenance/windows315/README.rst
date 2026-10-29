Windows Python 3.15 maintenance tools
===================================

The current Kraken checkout uses official PyTorch 2.15 and TorchVision 0.29.2
wheels for CPU and CUDA. See ``docs/user_guide/windows.rst``.

``build_torch315.py`` and ``assemble_torch315_wheel.py`` are historical tools
for reproducing the private PyTorch 2.14.1 runtime used by earlier local
builds. They are retained with their licenses and are not used for current
installations or by the manual workflow. Do not use their output with the
current Kraken dependency requirements.

``verify_shared_pylaia.py`` checks optional PyLaia 1.2.11+fork.20261028 with
the official PyTorch 2.15 runtime. It verifies both import orders, three CTC
training steps and a checkpoint round-trip. The older 1.2.8 bundle still pins
PyTorch 2.14.1.
