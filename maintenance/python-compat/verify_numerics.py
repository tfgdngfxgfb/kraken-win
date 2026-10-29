"""Check local SymPy/mpmath and optional cwevals without contacting a service."""

import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--check-cwevals', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    import kraken.models  # noqa: F401
    import mpmath as mp
    import sympy as sp
    import torch
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    assert importlib.metadata.version('sympy') == '1.14.0+pylaia1'
    assert importlib.metadata.version('mpmath') == '1.4.1'
    x = sp.Symbol('x')
    assert sp.simplify(sp.sin(x) ** 2 + sp.cos(x) ** 2) == 1
    assert sp.integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo)) == sp.sqrt(sp.pi)
    assert sp.Matrix([[1, 2], [3, 4]]).det() == -2
    with mp.workdps(80):
        numeric = sp.lambdify(x, sp.sin(x) ** 2 + sp.cos(x) ** 2, 'mpmath')
        assert mp.almosteq(numeric(mp.pi / 7), 1)
        assert abs(mp.mpf(str(sp.N(sp.sqrt(2), 75))) - mp.sqrt(2)) < mp.mpf('1e-74')
    shape = ShapeEnv().create_unbacked_symint()
    expression = ((shape + 1) * 2).node.expr
    assert sp.expand(expression - 2 * shape.node.expr) == 2
    devices = ['cpu', 'cuda'] if args.require_cuda else ['cpu']
    for device in devices:
        values = torch.linspace(0.1, 0.9, 8, device=device, requires_grad=True)
        output = (torch.sin(values) + values**3).sum()
        output.backward()
        torch.testing.assert_close(values.grad, torch.cos(values.detach()) + 3 * values.detach()**2)
        assert torch.isfinite(values.grad).all()
    result = {'sympy': sp.__version__, 'mpmath': mp.__version__,
              'symbolic_shapes': 'passed', 'high_precision_lambdify_and_integration': 'passed',
              'torch_gradients': devices}
    if args.check_cwevals:
        import httpx
        from coreweave_evaluations import Client, AsyncClient

        requests = []

        def response(request):
            assert request.url.host == 'example.invalid'
            requests.append(request.method)
            return httpx.Response(200, json={'version': 'mock-1', 'status': 'ok'})

        transport = httpx.MockTransport(response)
        with Client(api_key='offline-test', bearer_token='offline-test', base_url='https://example.invalid',
                    http_client=httpx.Client(transport=transport)) as client:
            assert client.service.get().model_dump()['status'] == 'ok'

        async def asynchronous():
            async with AsyncClient(api_key='offline-test', bearer_token='offline-test', base_url='https://example.invalid',
                                   http_client=httpx.AsyncClient(transport=transport)) as client:
                assert (await client.service.get()).model_dump()['status'] == 'ok'

        asyncio.run(asynchronous())
        assert requests == ['GET', 'GET']
        result['cwevals_mock_transport'] = 'sync and async passed; no network transport'
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
