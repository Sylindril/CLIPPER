# vendored wheels

## corels-1.1.29 (patched)

`corels` (pycorels 1.1.29, the certifiably-optimal rule-list learner) is the last
release on PyPI and does not build on modern Python/toolchains out of the box:

1. `setup.py` pokes `__builtins__.__NUMPY_SETUP__`, which is a `dict` (not a
   module) under PEP 517 build backends → `AttributeError`.
2. The shipped Cython-generated `corels/_corels.cpp` was produced by an old
   Cython and `#include "longintrepr.h"`, which Python 3.11 moved under
   `cpython/` → `fatal error: 'longintrepr.h' file not found`.

The wheel here is built from a locally-patched sdist. To rebuild (macOS/arm64,
Python 3.11, Homebrew `gmp` installed — `brew install gmp`):

```bash
# 1. fetch + unpack the sdist
url=$(curl -s https://pypi.org/pypi/corels/json | python3 -c \
  "import sys,json;d=json.load(sys.stdin);print([u['url'] for u in d['releases']['1.1.29'] if u['url'].endswith('.tar.gz')][0])")
mkdir -p /tmp/corels_build && curl -sL "$url" | tar xz -C /tmp/corels_build
cd /tmp/corels_build/corels-1.1.29

# 2. patch fix (1): drop the dict-unsafe __NUMPY_SETUP__ line in setup.py
sed -i '' '/__NUMPY_SETUP__/d' setup.py

# 3. patch fix (2): regenerate _corels.cpp with a modern Cython
rm -f corels/_corels.cpp
uv run --no-project python -c \
  "import sys;sys.argv=['cython','--cplus','-3','corels/_corels.pyx'];from Cython.Compiler.Main import setuptools_main;sys.exit(setuptools_main())"

# 4. build the wheel with Homebrew gmp on the search path
GMP=$(brew --prefix gmp); export CPATH="$GMP/include" LIBRARY_PATH="$GMP/lib"
export VIRTUAL_ENV=/path/to/this/repo/.venv
uv pip install --no-build-isolation .   # or build a wheel; then copy it into vendor/
```

The venv needs `setuptools wheel Cython numpy` for the no-isolation build
(`uv pip install setuptools wheel Cython`). The resulting extension links
`libgmp` dynamically, so Homebrew `gmp` must stay installed at runtime.

### Windows rebuild (corels-1.1.29-cp311-cp311-win_amd64.whl)

Same two patches, MSVC toolchain (VS 2022 C++ tools; setuptools finds it via
vswhere), **no GMP**: `setup.py` retries `install(False)` when the gmp link
fails, and the upstream C++ guards GMP behind `#ifdef GMP`, so the Windows
extension builds without it. From Git Bash, in an unpacked sdist:

```bash
sed -i '/__NUMPY_SETUP__/d' setup.py
rm -f corels/_corels.cpp
uv run --no-project --with cython,numpy cython --cplus -3 corels/_corels.pyx
uv run --no-project --python 3.11 --with setuptools,wheel,numpy \
    python setup.py bdist_wheel        # -> dist/corels-1.1.29-cp311-cp311-win_amd64.whl
```

Runtime caveats both wheels share (handled in `scripts/imodels_baselines.py`,
not the wheel): pycorels does `np.array(x, copy=False)` internally, which
NumPy 2 forbids when a copy is needed — feed it contiguous uint8 ndarrays —
and its C extension encodes feature names as ASCII, so transliterate names
from non-ASCII vocabularies.
