import sys

__version__ = "0.2.0"

# The code uses Python 3.12 syntax (nested quotes in f-strings). Fail with a clear message instead of a SyntaxError.
if sys.version_info < (3, 12):
    raise SystemExit(f"Blocky needs Python 3.12 or newer (this is {sys.version.split()[0]}). "
                     "Ubuntu 24.04 ships 3.12; on older systems install a newer python3 first.")
