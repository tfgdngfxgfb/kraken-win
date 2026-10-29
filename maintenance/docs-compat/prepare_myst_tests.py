"""Copy upstream tests and update one Docutils 0.23 diagnostic snapshot.

Docutils now retains the already reported implicit-target override INFO message
in the document tree. Keep that message visible and check it in the expected
tree; reference targets and the existing stderr assertions remain unchanged.
"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.source_dir / 'tests', args.output_dir / 'tests')
    shutil.copy2(args.source_dir / 'pyproject.toml', args.output_dir / 'pyproject.toml')
    fixture = args.output_dir / 'tests/test_renderers/fixtures/docutil_link_resolution.md'
    original = fixture.read_text(encoding='utf-8')
    before = '    <subtitle ids="other test-1" names="other test">\n        Other\n    <target refid="test-1">'
    after = ('    <subtitle ids="other test-1" names="other test">\n        Other\n'
             '    <system_message level="1" line="3" source="<src>/index.md" type="INFO">\n'
             '        <paragraph>\n'
             '            Target name overrides implicit target name "test".\n'
             '    <target refid="test-1">')
    if original.count(before) != 1:
        raise ValueError('Unexpected upstream target-override snapshot.')
    updated = original.replace(before, after, 1)
    fixture.write_text(updated, encoding='utf-8')
    patch = ''.join(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                       fromfile='upstream/docutil_link_resolution.md',
                                       tofile='docutils023/docutil_link_resolution.md'))
    (args.output_dir / 'docutils023-fixture.patch').write_text(patch, encoding='utf-8')
    (args.output_dir / 'TEST-ADAPTATION.json').write_text(json.dumps({
        'fixture': str(fixture.relative_to(args.output_dir)),
        'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'updated_sha256': hashlib.sha256(updated.encode()).hexdigest(),
        'reason': 'Retain and assert Docutils 0.23 INFO node for explicit-over-implicit target names.',
        'production_code_changed': False, 'assertions_removed': False,
    }, indent=2) + '\n', encoding='utf-8')
    print(patch)


if __name__ == '__main__':
    main()
