"""Exercise Kraken and PyLaia training in a shared Python environment.

Run in separate processes for each import order so normal CUDA shutdown is also
checked by the caller. This optional check does not make PyLaia a Kraken dependency.
"""

import argparse
import importlib.metadata
import io
import json
from pathlib import Path
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--import-order', choices=('kraken-first', 'laia-first'), required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.import_order == 'laia-first':
        import laia  # noqa: F401
        import kraken.models
    else:
        import kraken.models
        import laia  # noqa: F401

    import torch
    from kraken.tasks import SegmentationTaskModel
    from laia.models.htr import LaiaCRNN

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA validation requested, but CUDA is unavailable.')

    def make_model():
        return LaiaCRNN(
            num_input_channels=1, num_output_labels=4, cnn_num_features=[4],
            cnn_kernel_size=[3], cnn_stride=[1], cnn_dilation=[1],
            cnn_activation=[torch.nn.LeakyReLU], cnn_poolsize=[2],
            cnn_dropout=[0.1], cnn_batchnorm=[False], image_sequencer='maxpool-1',
            rnn_units=8, rnn_layers=2, rnn_dropout=0.5, lin_dropout=0.1)

    segmenter = SegmentationTaskModel.load_model()
    results = {}
    for device in (['cpu', 'cuda'] if args.require_cuda else ['cpu']):
        torch.manual_seed(20261017)
        model = make_model().to(device)
        before = model.linear.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        images = torch.randn(2, 1, 16, 32, device=device)
        targets = torch.tensor([1, 2, 2, 1], device=device, dtype=torch.long)
        lengths = torch.tensor([2, 2], dtype=torch.long)
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            output = model(images)
            time_steps = torch.full((2,), output.shape[0], dtype=torch.long)
            loss = torch.nn.functional.ctc_loss(output.log_softmax(2), targets, time_steps, lengths)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            losses.append(loss.item())
        assert not torch.equal(before, model.linear.weight.detach())
        state = io.BytesIO()
        torch.save(model.state_dict(), state)
        state.seek(0)
        restored = make_model().to(device)
        restored.load_state_dict(torch.load(state, map_location=device, weights_only=True))
        model.eval()
        restored.eval()
        with torch.inference_mode():
            torch.testing.assert_close(model(images), restored(images))
        assert model.rnn.dropout == 0.5 and torch.backends.cudnn.enabled
        results[device] = {'ctc_steps': 3, 'finite_losses': losses,
                           'weights_updated': True, 'checkpoint_roundtrip': 'passed'}

    report = {'python': platform.python_version(), 'import_order': args.import_order,
              'kraken': importlib.metadata.version('kraken'),
              'kraken_module': str(Path(kraken.models.__file__).resolve()),
              'pylaia': importlib.metadata.version('pylaia'),
              'pillow': importlib.metadata.version('Pillow'),
              'filelock': importlib.metadata.version('filelock'),
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name() if args.require_cuda else None,
              'standard_segmentation_model_loaded': segmenter is not None,
              'cudnn_enabled': torch.backends.cudnn.enabled, 'training': results}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
