Local SymPy and optional W&B compatibility pairs
================================================

The supplied PyLaia 1.2.11 Windows bundle provides SymPy 1.14.0+pylaia1 with
mpmath 1.4.1, and optional W&B 0.31.0+pylaia2 with cwevals 0.0.1b3.
Only dependency metadata and version strings differ from upstream. Numerical
code and W&B's native backend remain byte-identical to the official packages.
No PyTorch binding files are changed.

build_compatibility.py is adapted from that bundle under the accompanying
MIT license. provenance.json records source hashes, adaptations and supplied
artifacts. All compatibility wheels were reproduced byte-identically.
The builder checks pinned artifacts and RECORD hashes and preserves licenses,
native payloads and executable permissions.

For a Kraken source installation without the PyLaia Python package::

   python maintenance/python-compat/build_compatibility.py --sympy-only --output .local/python-compat-wheels
   python -m pip install --find-links .local/python-compat-wheels -c maintenance/dependency-constraints.txt -e ".[test,pdf]"

For documentation, build the local MyST wheel described in
maintenance/docs-compat/README.rst, then add its folder to pip::

   python -m pip install --find-links .local/python-compat-wheels --find-links .local/docs-compat-wheels -c maintenance/dependency-constraints.txt -e ".[test,pdf,docs]"

A shared PyLaia environment uses its supplied wheel and both compatibility
and wheels/cp315 directories as find-links locations. W&B is optional; its
checks use offline mode. cwevals checks use mock transports without service
calls. Both programs use unmodified official PyTorch 2.15 CPU/CUDA wheels.

Omitting --sympy-only builds the full compatibility set, including optional
W&B, without installing it. verify_numerics.py checks high-precision arithmetic,
symbolic shapes/gradients and synchronous/asynchronous cwevals mock clients.

provenance-20261020.json retains the older PyLaia 1.2.8 compatibility record.
