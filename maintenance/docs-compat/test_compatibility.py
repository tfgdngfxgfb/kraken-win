"""Integration checks for the private Sphinx/MyST/Docutils combination."""

import importlib.metadata
import subprocess
import sys

import pytest


@pytest.mark.parametrize('builder,pattern', [('html', 'page.html'), ('text', 'page.txt'), ('latex', '*.tex')])
def test_mixed_rst_markdown_build(tmp_path, builder, pattern):
    source = tmp_path / 'source'
    output = tmp_path / 'output'
    source.mkdir()
    (source / 'conf.py').write_text(
        "project='Compatibility'\nextensions=['myst_parser','sphinx.ext.mathjax']\n"
        "myst_enable_extensions=['colon_fence','deflist','dollarmath']\n"
        "source_suffix={'.rst':'restructuredtext','.md':'markdown'}\n"
        "latex_documents=[('index','compat.tex','Compatibility','Kraken','manual')]\n",
        encoding='utf-8')
    (source / 'index.rst').write_text(
        'Compatibility\n=============\n\n'
        '.. toctree::\n\n   page\n\n'
        'See :ref:`compat-anchor`.\n\n'
        '.. csv-table:: Values\n   :header: "Name", "Value"\n\n   "one", "1"\n\n'
        '.. include:: included.rst\n\n'
        '.. literalinclude:: sample.py\n   :language: python\n', encoding='utf-8')
    (source / 'included.rst').write_text('Included **reStructuredText**.\n', encoding='utf-8')
    (source / 'sample.py').write_text('value = 42\n', encoding='utf-8')
    (source / 'page.md').write_text(
        '(compat-anchor)=\n# Kompatibilitet æøå\n\n'
        'A paragraph with **bold**, *italic* and a footnote.[^note]\n\n'
        '[^note]: Footnote compatibility.\n\n'
        '::: {note}\nA directive from Markdown.\n:::\n\n'
        'Term\n: Definition\n\n'
        '$$\nx^2 + y^2 = z^2\n$$\n', encoding='utf-8')
    result = subprocess.run(
        [sys.executable, '-I', '-m', 'sphinx', '-b', builder, '-W', '--keep-going',
         '-D', 'exclude_patterns=included.rst', str(source), str(output)],
        capture_output=True, text=True, encoding='utf-8', timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    products = list(output.glob(pattern))
    assert products, result.stdout
    rendered = '\n'.join(path.read_text(encoding='utf-8') for path in products)
    assert 'Kompatibilitet' in rendered
    assert 'Footnote compatibility' in rendered


def test_direct_myst_docutils_html():
    from docutils.core import publish_string
    from myst_parser.parsers.docutils_ import Parser

    assert importlib.metadata.version('docutils') == '0.23'
    assert importlib.metadata.version('Sphinx') == '9.2.0'
    assert importlib.metadata.version('myst-parser') == '5.1.0+kraken2'
    result = publish_string('# Direct Markdown\n\nA [link](https://example.invalid).\n',
                            parser=Parser(), writer='html5',
                            settings_overrides={'halt_level': 2}).decode('utf-8')
    assert 'Direct Markdown' in result
    assert 'https://example.invalid' in result


def test_explicit_target_overrides_implicit_heading():
    from io import StringIO
    from docutils import nodes
    from docutils.core import publish_doctree
    from myst_parser.parsers.docutils_ import Parser

    messages = StringIO()
    tree = publish_doctree('# Test\n\n(test)=\n## Other\n\n[](#test)\n',
                           source_path='<src>/index.md', parser=Parser(),
                           settings_overrides={'myst_heading_anchors': 1, 'report_level': 0,
                                               'warning_stream': messages})
    references = list(tree.findall(nodes.reference))
    assert len(references) == 1 and references[0]['refid'] == 'test-1'
    assert references[0].astext() == 'Other'
    diagnostics = list(tree.findall(nodes.system_message))
    assert any(node['level'] == 1 and 'Target name overrides implicit target name' in node.astext()
               for node in diagnostics)
    assert 'Target name overrides implicit target name "test".' in messages.getvalue()
