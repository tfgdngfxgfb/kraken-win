Local MyST build for official Sphinx 9.2 and Docutils 0.23
==========================================================

The current stack uses official Sphinx 9.2.0 and Docutils 0.23 with local
MyST-Parser 5.1.0+kraken2. Sphinx now supports Docutils 0.23 directly; its
previous private build is no longer needed. MyST still needs its older
Docutils restriction updated.

MyST changes are limited to dependency metadata, its local version string
and source build metadata for flit-core 4.2.0. Its Python implementation and
MIT license remain unchanged. The test extra accepts Pygments 2.21.

Build the local wheel and source archive from SHA-256-pinned upstream files::

   python maintenance/docs-compat/build_compat_wheels.py --cache-dir .local/docs-upstream --output-dir .local/docs-compat-wheels
   python -m pip install --find-links .local/docs-compat-wheels -c maintenance/dependency-constraints.txt "sphinx==9.2.0" "myst-parser==5.1.0+kraken2" "docutils==0.23"

The builder checks original RECORD hashes, preserves licenses and file
permissions, and refuses to replace a different build. The source archive
rebuilds with current Flit; its runtime package files match the tested wheel.

Local integration checks cover RST/Markdown documents, cross-references,
footnotes, includes, CSV tables, code, mathematics and HTML/text/LaTeX output::

   python -m pytest maintenance/docs-compat/test_compatibility.py

The full MyST upstream suite uses the same documented Docutils 0.23 snapshot
adaptation as before. Docutils retains an INFO node when an explicit target
overrides an implicit heading target. prepare_myst_tests.py preserves and
asserts that node; it does not hide diagnostics or remove assertions.

provenance.json records the current stack and artifacts. The earlier private
Sphinx 9.1.1/MyST kraken1 provenance is retained in provenance-20261020.json.
Earlier transfer archives are unchanged. No CI or publishing is performed.
