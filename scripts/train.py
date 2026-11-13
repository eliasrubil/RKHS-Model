"""Train and evaluate every model.

Thin wrapper around :mod:`rkhs_credit.train` that works without installing the
package::

    python scripts/train.py --config configs/default.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkhs_credit.train import main  # noqa: E402

if __name__ == "__main__":
    main()
