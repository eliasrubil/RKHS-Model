"""Generate every figure of the report into ``docs/figures``.

Thin wrapper around :mod:`rkhs_credit.make_figures` that works without
installing the package (run ``scripts/train.py`` first)::

    python scripts/make_figures.py --config configs/default.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkhs_credit.make_figures import main  # noqa: E402

if __name__ == "__main__":
    main()
