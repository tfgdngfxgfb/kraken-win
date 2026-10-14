from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from skimage.measure import label, regionprops

from kraken.configs import RecognitionInferenceConfig, SegmentationInferenceConfig
from kraken.containers import BBoxLine, Segmentation
from kraken.lib.arrow_dataset import build_binary_dataset
from kraken.lib.segmentation import boundary_tracing
from kraken.models import create_model
from kraken.tasks import RecognitionTaskModel

RESOURCES = Path(__file__).resolve().parent / 'resources'


@pytest.mark.parametrize('mask', [np.ones((1, 1)), np.ones((1, 4)), np.eye(4), np.ones((3, 3))])
def test_boundary_tracing_small_regions(mask):
    region = regionprops(label(mask, connectivity=2))[0]
    boundary = boundary_tracing(region)
    assert len(boundary) > 0
    assert all(tuple(point) in set(map(tuple, region.coords)) for point in boundary)


def test_curved_baseline_polygon_extraction():
    from helpers import load_segmentation
    from kraken.lib.segmentation import extract_polygons

    segmentation = load_segmentation('arabic_seg.json')
    segmentation.lines = segmentation.lines[:1]
    with Image.open(RESOURCES / 'arabic.webp') as im:
        crop, _ = next(extract_polygons(im, segmentation, legacy=False))
    assert crop.width > 0 and crop.height > 0
    assert np.ptp(np.asarray(crop)) > 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@torch.inference_mode()
def test_segmentation_input_device(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    model = create_model('TorchVGSLModel',
                         vgsl='[1,32,0,1 Cr3,3,4 O2l4]',
                         model_type=['segmentation'],
                         class_mapping={'aux': {'start': 0, 'end': 1},
                                        'baselines': {'default': 2},
                                        'regions': {'text': 3}})
    model.prepare_for_inference(SegmentationInferenceConfig(accelerator=device, device=1))
    heatmap = model._compute_segmentation_map(Image.new('L', (64, 32), 255))['heatmap']
    assert heatmap.shape[0] == 4
    assert np.isfinite(heatmap).all()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_recognition_input_device(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    task = RecognitionTaskModel.load_model(RESOURCES / 'overfit.mlmodel')
    with Image.open(RESOURCES / '000236.png') as im:
        seg = Segmentation(type='bbox', imagename=im.filename,
                           text_direction='horizontal-lr', script_detection=False,
                           lines=[BBoxLine(id='line', bbox=(0, 0, im.width, im.height))])
        config = RecognitionInferenceConfig(accelerator=device, device=1, num_line_workers=0)
        records = list(task.predict(im, seg, config))
    assert len(records) == 1
    assert records[0].prediction


def test_compile_dataset_spawn_workers(tmp_path):
    build_binary_dataset(files=[RESOURCES / '170025120000003,0074-lite.xml'],
                         output_file=str(tmp_path / 'lines.arrow'),
                         format_type='xml', num_workers=2)
    import pyarrow as pa
    with pa.OSFile(str(tmp_path / 'lines.arrow'), 'rb') as source:
        assert pa.ipc.open_file(source).read_all().num_rows == 4


def test_xml_image_path_relative_to_output(tmp_path):
    from click.testing import CliRunner
    from kraken.kraken import cli
    from kraken.lib.xml import XMLPage

    image = tmp_path / 'images' / 'page.png'
    output = tmp_path / 'segments' / 'page.xml'
    image.parent.mkdir()
    output.parent.mkdir()
    with Image.open(RESOURCES / 'input.webp') as im:
        im.save(image)
    result = CliRunner().invoke(cli, ['-r', '-x', '-i', str(image), str(output), 'segment', '-bl'])
    assert result.exit_code == 0, result.output
    assert Path(XMLPage(output).imagename).resolve() == image.resolve()


def test_class_remapping_rebuilds_training_targets():
    from helpers import make_baseline_segmentation
    from kraken.containers import Region
    from kraken.lib.dataset.segmentation import BaselineSet

    aux = {'_start_separator': 0, '_end_separator': 1}
    original = {'aux': aux, 'baselines': {'line': 3}, 'regions': {'text': 2}}
    remapped = {'aux': aux, 'baselines': {'line': 2}, 'regions': {'text': 3}}
    doc = make_baseline_segmentation(tags={'type': [{'type': 'line'}]})
    doc.regions = {'text': [Region(id='region', boundary=[[0, 0], [100, 0], [100, 20], [0, 20]])]}
    dataset = BaselineSet(class_mapping=original)
    dataset.add(doc)
    dataset.set_class_mapping(remapped)
    assert set(dataset.targets[0]['baselines']) == {2}
    assert set(dataset.targets[0]['regions']) == {3}
    assert dataset.class_stats['baselines']['line'] == 1
    assert dataset.class_stats['regions']['text'] == 1
    assert dataset.canonical_class_mapping == remapped


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_pdf_segmentation(tmp_path, device):
    pytest.importorskip('pypdfium2')
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    import json
    from click.testing import CliRunner
    from kraken.kraken import cli

    pdf = tmp_path / 'input.pdf'
    with Image.open(RESOURCES / 'input.webp') as im:
        im.convert('RGB').resize((300, 450)).save(pdf, 'PDF')
    result = CliRunner().invoke(cli, ['-r', '-d', device, '-f', 'pdf', '-I', str(pdf),
                                      '-o', '.json', '-p', str(tmp_path / 'page_{idx}'),
                                      'segment', '-bl'])
    assert result.exit_code == 0, result.output
    result_data = json.loads((tmp_path / 'page_0.json').read_text(encoding='utf-8'))
    assert result_data['type'] == 'baselines'
    assert result_data['lines']


def test_binarize_xml_output(tmp_path):
    from click.testing import CliRunner
    from kraken.kraken import cli
    from kraken.lib.xml import XMLPage

    output = tmp_path / 'page.xml'
    result = CliRunner().invoke(cli, ['-r', '-x', '-i', str(RESOURCES / 'input.webp'),
                                      str(output), 'binarize'])
    assert result.exit_code == 0, result.output
    assert Path(XMLPage(output).imagename).is_file()


def test_segtest_non_contiguous_pixel_channels():
    from click.testing import CliRunner
    from kraken.ketos import cli

    result = CliRunner().invoke(cli, ['-d', 'cpu', '--threads', '1', '--workers', '0',
                                      'segtest', '-m', str(RESOURCES.parent.parent / 'kraken' / 'blla.mlmodel'),
                                      str(RESOURCES / '170025120000003,0074-lite.xml')])
    assert result.exit_code == 0, result.output
    assert 'Pixel Accuracy' in result.output


def test_finetuning_mapping_checkpoint(tmp_path):
    from helpers import make_baseline_segmentation, make_smoke_trainer
    from kraken.containers import Region
    from kraken.configs import BLLASegmentationTrainingConfig, BLLASegmentationTrainingDataConfig
    from kraken.train import BLLASegmentationModel, BLLASegmentationDataModule

    old_mapping = {'aux': {'_start_separator': 0, '_end_separator': 1},
                   'baselines': {'line': 3}, 'regions': {'text': 2}}
    net = create_model('TorchVGSLModel', vgsl='[1,32,0,1 Cr3,3,4 O2l4]',
                       model_type=['segmentation'], class_mapping=old_mapping)
    doc = make_baseline_segmentation(tags={'type': [{'type': 'line'}]})
    doc.regions = {'text': [Region(id='region', boundary=[[0, 0], [100, 0], [100, 20], [0, 20]])]}
    data_config = BLLASegmentationTrainingDataConfig(format_type=None,
                                                     training_data=[{'doc': doc}],
                                                     evaluation_data=[{'doc': doc}], num_workers=0)
    datamodule = BLLASegmentationDataModule(data_config)
    model = BLLASegmentationModel(BLLASegmentationTrainingConfig(), model=net)
    trainer = make_smoke_trainer(limit_train_batches=1, limit_val_batches=1)
    trainer.fit(model, datamodule)
    checkpoint = tmp_path / 'mapping.ckpt'
    trainer.save_checkpoint(checkpoint)
    loaded = BLLASegmentationModel.load_from_checkpoint(checkpoint, weights_only=False)
    assert loaded.net.user_metadata['class_mapping'] == old_mapping
    assert loaded.net.user_metadata['_full_class_mapping'] == old_mapping
    assert set(datamodule.train_set.dataset.targets[0]['baselines']) == {3}
    assert set(datamodule.train_set.dataset.targets[0]['regions']) == {2}
