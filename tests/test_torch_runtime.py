"""Exercise official bindings and the cuDNN cleanup needed for normal shutdown."""
import os
import subprocess
import sys

import pytest
import torch


@pytest.mark.skipif(sys.platform != 'win32' or sys.version_info[:2] != (3, 15),
                    reason='Windows Python 3.15 binding regression')
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('with_kraken', [False, True], ids=['official-api-cleanup', 'with-kraken'])
def test_autograd_and_normal_shutdown(device, with_kraken):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    code = '''
import ctypes
import sys
import torch

if sys.argv[2] == '1':
    import kraken.models
elif sys.argv[1] == 'cuda':
    # The official 2.15 bindings fix autograd, but cuDNN's cached dropout state
    # still needs clearing before DLL unload. Exercise its API without Kraken.
    import atexit
    atexit.register(torch._C._cudnn_clear_dropout_state)

get_dict = ctypes.pythonapi.PyThreadState_GetDict
get_dict.restype = ctypes.c_void_p
get_dict.argtypes = []

class Identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        pointer = get_dict()
        assert pointer and pointer % ctypes.alignment(ctypes.py_object) == 0
        return gradient

device = sys.argv[1]
value = torch.ones(2, device=device, requires_grad=True)
Identity.apply(value).sum().backward()
assert value.grad.tolist() == [1.0, 1.0]
if device == 'cuda':
    model = torch.nn.LSTM(16, 16, 2, bidirectional=True, dropout=0.5).cuda()
    inputs = torch.randn(4, 2, 16, device=device, requires_grad=True)
    model(inputs)[0].sum().backward()
    assert inputs.grad is not None and model.dropout == 0.5
    assert torch.backends.cudnn.enabled
print('autograd and dropout verified', flush=True)
'''
    result = subprocess.run([sys.executable, '-I', '-X', 'faulthandler', '-c', code, device, str(int(with_kraken))],
                            capture_output=True, text=True, encoding='utf-8', timeout=90,
                            env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'autograd and dropout verified' in result.stdout
