"""Windows Python 3.15 compatibility for PyTorch 2.14.1.

The shutdown cleanup is adapted from the supplied PyLaia 1.2.3 fork.
See PYLAIA-LICENSE.txt for its MIT license.
"""
import atexit
import sys

import torch

_configured = False


def configure_windows_torch():
    """Check the binding ABI and release cuDNN dropout state before DLL unload."""
    global _configured
    if _configured or sys.platform != 'win32' or sys.version_info[:2] != (3, 15):
        return
    if str(torch.__version__).split('+')[0] == '2.14.1':
        if getattr(torch.version, '_pylaia_python315_abi', None) != 1:
            raise RuntimeError(
                'Windows Python 3.15 needs the corrected PyTorch 2.14.1 bindings. '
                'The official wheels have an incompatible Python autograd ABI. '
                'Use the supplied PyLaia 3.15 runtime or the binding builder in '
                'maintenance/windows315; see docs/user_guide/windows.rst.')
        cleanup = getattr(torch._C, '_cudnn_clear_dropout_state', None)
        if torch.version.cuda is not None and cleanup is not None:
            is_initialized = torch.cuda.is_initialized

            def release_dropout_state():
                if is_initialized():
                    cleanup()

            atexit.register(release_dropout_state)
    _configured = True
