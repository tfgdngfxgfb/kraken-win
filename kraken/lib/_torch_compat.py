"""Use official Windows Python 3.15 bindings and clean up cuDNN before unload.

The shutdown cleanup follows the earlier MIT-licensed PyLaia integration.
See PYLAIA-LICENSE.txt. No native binding files are modified.
"""
import atexit
import sys

import torch
from packaging.version import Version


_configured = False


def configure_windows_torch():
    """Require PyTorch 2.15+ and release its official cuDNN dropout cache at exit."""
    global _configured
    if _configured or sys.platform != 'win32' or sys.version_info[:2] != (3, 15):
        return
    if Version(torch.__version__).release < (2, 15):
        raise RuntimeError(
            'Windows Python 3.15 requires official PyTorch 2.15 or newer. '
            'Update torch and torchvision from the official CPU or CUDA index; '
            'see docs/user_guide/windows.rst. The old private PyLaia 2.14.1 '
            'runtime is not supported by this Kraken version.')
    cleanup = getattr(torch._C, '_cudnn_clear_dropout_state', None)
    if torch.version.cuda is not None and callable(cleanup):
        is_initialized = torch.cuda.is_initialized

        def release_dropout_state():
            if is_initialized():
                cleanup()

        atexit.register(release_dropout_state)
    _configured = True
