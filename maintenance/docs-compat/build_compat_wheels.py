"""Build private MyST wheels and source archives for official Sphinx 9.2 and Docutils 0.23.

Only the SHA-256-pinned upstream artifacts below are accepted. Native code and
PyTorch are outside this builder. Licenses and upstream sources are preserved.
"""

import argparse
import base64
import copy
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import urllib.request
import zipfile


RECIPES = (
    {
        'name': 'myst_parser', 'version': '5.1.0', 'local': '5.1.0+kraken2',
        'wheel_sha256': '9c91c52b3cdb4d94a6506e4fab4e2f296c7623a0da0dcbe6de1565c3dad67a8a',
        'sdist_sha256': 'ab69322dc6719dcc7f296479dbb70181b66df6ed315064f92dbc85c0e1bf2f02',
        'version_file': 'myst_parser/__init__.py',
        'version_old': '__version__ = "5.1.0"',
        'version_new': '__version__ = "5.1.0+kraken2"',
        'source_requirements': [('docutils>=0.20,<0.23', 'docutils==0.23'),
                                ('sphinx>=8,<10', 'sphinx==9.2.0'),
                                ('flit_core >=3.4,<4', 'flit_core >=4.2,<5'),
                                ('pygments<2.21', 'pygments>=2.21,<3')],
    },
)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe archive name: {name}')


def fetch(recipe, kind, cache):
    name, version = recipe['name'], recipe['version']
    filename = f'{name}-{version}-py3-none-any.whl' if kind == 'wheel' else f'{name}-{version}.tar.gz'
    expected = recipe[kind + '_sha256']
    target = cache / filename
    if not target.exists():
        with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=35) as stream:
            metadata = json.load(stream)
        entry = next(item for item in metadata['urls'] if item['filename'] == filename)
        if entry['digests']['sha256'] != expected:
            raise ValueError('Upstream artifact hash differs from pinned provenance.')
        with urllib.request.urlopen(entry['url'], timeout=60) as stream:
            content = stream.read()
        if digest(content) != expected:
            raise ValueError('Downloaded artifact failed SHA-256 verification.')
        target.write_bytes(content)
    content = target.read_bytes()
    if digest(content) != expected:
        raise ValueError(f'Cached artifact failed SHA-256 verification: {target}')
    return content


def replace_once(content, old, new):
    old, new = old.encode(), new.encode()
    if content.count(old) != 1:
        raise ValueError(f'Expected exactly one replacement: {old!r}')
    return content.replace(old, new, 1)


def patch_code(name, content, recipe):
    if name == recipe['version_file']:
        content = replace_once(content, recipe['version_old'], recipe['version_new'])
    return content


def patch_metadata(content, recipe):
    text = content.decode('utf-8')
    text, count = re.subn(r'^Version: [^\r\n]+', f"Version: {recipe['local']}", text, count=1, flags=re.M)
    if count != 1:
        raise ValueError('Missing package version header.')
    text, count = re.subn(r'^Requires-Dist: docutils[^\r\n]+', 'Requires-Dist: docutils==0.23', text, flags=re.M)
    if count != 1:
        raise ValueError('Unexpected Docutils requirements.')
    if recipe['name'] == 'myst_parser':
        text, count = re.subn(r'^Requires-Dist: sphinx[^\r\n]+(?<!["\'])$',
                              'Requires-Dist: sphinx==9.2.0', text, count=1, flags=re.M)
        if count != 1:
            raise ValueError('Missing MyST runtime Sphinx requirement.')
        if text.count('Requires-Dist: pygments<2.21') != 1:
            raise ValueError('Unexpected MyST test Pygments requirement.')
        text = text.replace('Requires-Dist: pygments<2.21', 'Requires-Dist: pygments<3,>=2.21', 1)
    return text.encode('utf-8')


def write_new(path, content):
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f'Refusing to replace a different build: {path}')
    path.write_bytes(content)


def wheel(recipe, content, output):
    old = f"{recipe['name']}-{recipe['version']}.dist-info"
    new = f"{recipe['name']}-{recipe['local']}.dist-info"
    with zipfile.ZipFile(io.BytesIO(content)) as source:
        if len(source.namelist()) != len(set(source.namelist())):
            raise ValueError('Duplicate upstream wheel entries.')
        for name in source.namelist():
            safe_name(name)
        rows = csv.reader(io.StringIO(source.read(old + '/RECORD').decode()))
        checked = set()
        for name, expected, size in rows:
            checked.add(name)
            data = source.read(name)
            if expected:
                actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
                if expected != 'sha256=' + actual or int(size) != len(data):
                    raise ValueError(f'Invalid upstream RECORD entry: {name}')
        if checked != {name for name in source.namelist() if not name.endswith('/')}:
            raise ValueError('Upstream wheel files differ from RECORD.')
        files = {}
        for info in source.infolist():
            if info.is_dir() or info.filename == old + '/RECORD':
                continue
            data = source.read(info)
            name = info.filename.replace(old + '/', new + '/', 1)
            if name == new + '/METADATA':
                data = patch_metadata(data, recipe)
            else:
                data = patch_code(name, data, recipe)
            files[name] = (data, info.external_attr)
    record = io.StringIO(newline='')
    writer = csv.writer(record, lineterminator='\n')
    for name, (data, _) in sorted(files.items()):
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
        writer.writerow((name, 'sha256=' + encoded, len(data)))
    writer.writerow((new + '/RECORD', '', ''))
    files[new + '/RECORD'] = (record.getvalue().encode(), 0o100644 << 16)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for name, (data, permissions) in sorted(files.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = permissions
            target.writestr(info, data)
    filename = f"{recipe['name']}-{recipe['local']}-py3-none-any.whl"
    data = stream.getvalue()
    write_new(output / filename, data)
    return {'filename': filename, 'sha256': digest(data), 'upstream_sha256': recipe['wheel_sha256']}


def sdist(recipe, content, output):
    old = f"{recipe['name']}-{recipe['version']}"
    new = f"{recipe['name']}-{recipe['local']}"
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as target, tarfile.open(fileobj=io.BytesIO(content)) as source:
            for member in source:
                safe_name(member.name)
                if not member.isdir() and not member.isfile():
                    raise ValueError('Unexpected source archive link/device.')
                info = copy.copy(member)
                # A copied PAX path overrides TarInfo.name for long filenames.
                # Let tarfile regenerate it after changing the archive root.
                info.pax_headers = dict(member.pax_headers)
                info.pax_headers.pop('path', None)
                relative = member.name.removeprefix(old + '/')
                info.name = member.name.replace(old, new, 1)
                if member.isdir():
                    target.addfile(info)
                    continue
                data = source.extractfile(member).read()
                if relative == 'PKG-INFO':
                    data = patch_metadata(data, recipe)
                elif relative == 'pyproject.toml':
                    for before, after in recipe['source_requirements']:
                        data = replace_once(data, before, after)
                else:
                    data = patch_code(relative, data, recipe)
                info.size = len(data)
                target.addfile(info, io.BytesIO(data))
    filename = f"{recipe['name']}-{recipe['local']}.tar.gz"
    data = stream.getvalue()
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        if {name.split('/')[0] for name in archive.getnames()} != {new}:
            raise ValueError('Local source archive has more than one root directory.')
    write_new(output / filename, data)
    return {'filename': filename, 'sha256': digest(data), 'upstream_sha256': recipe['sdist_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'docutils': '0.23', 'local_only': True, 'published': False, 'artifacts': []}
    for recipe in RECIPES:
        manifest['artifacts'].append(wheel(recipe, fetch(recipe, 'wheel', args.cache_dir), args.output_dir))
        manifest['artifacts'].append(sdist(recipe, fetch(recipe, 'sdist', args.cache_dir), args.output_dir))
    (args.output_dir / 'BUILD-INFO.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
