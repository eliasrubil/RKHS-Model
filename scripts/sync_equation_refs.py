"""Keep the equation numbers quoted in code comments in sync with the report.

Mathematical comments cite the appendix as ``Ec. (A.12) [eq:some-label]``.  The
label is the stable anchor (it is the ``\\label`` used in ``docs/report.tex``);
the number is what the reader sees in the PDF and changes whenever an equation
is inserted.  After recompiling the report, run::

    python scripts/sync_equation_refs.py            # rewrite the numbers
    python scripts/sync_equation_refs.py --check    # only verify (exit code 1 on mismatch)

The numbers are read from ``docs/report.aux``, which ``latexmk`` produces.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = re.compile(r"Ec\. \((?P<number>[^)]*)\) \[(?P<label>eq:[A-Za-z0-9:_-]+)\]")
NEWLABEL = re.compile(r"\\newlabel\{(?P<label>eq:[^}]+)\}\{\{(?P<number>[^}]*)\}")


def read_equation_numbers(aux_path: Path) -> dict[str, str]:
    """Map every ``eq:...`` label of the compiled report to its printed number."""
    if not aux_path.exists():
        sys.exit(
            f"{aux_path} not found: compile the report first (cd docs && latexmk -pdf report.tex)."
        )
    text = aux_path.read_text(encoding="utf-8", errors="replace")
    return {match["label"]: match["number"] for match in NEWLABEL.finditer(text)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aux", default=str(REPO_ROOT / "docs" / "report.aux"))
    parser.add_argument("--check", action="store_true", help="Report mismatches, change nothing.")
    args = parser.parse_args()

    numbers = read_equation_numbers(Path(args.aux))
    problems = 0
    n_references = 0
    for folder in ("src", "tests", "scripts"):
        for path in sorted((REPO_ROOT / folder).rglob("*.py")):
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")

            def replace(match: re.Match, path: Path = path) -> str:
                nonlocal problems, n_references
                n_references += 1
                label = match["label"]
                if label not in numbers:
                    problems += 1
                    print(f"{path.relative_to(REPO_ROOT)}: unknown label [{label}]")
                    return match[0]
                if match["number"] != numbers[label]:
                    if args.check:
                        problems += 1
                        print(
                            f"{path.relative_to(REPO_ROOT)}: [{label}] is cited as "
                            f"({match['number']}) but the report says ({numbers[label]})"
                        )
                        return match[0]
                    return f"Ec. ({numbers[label]}) [{label}]"
                return match[0]

            new_text = REFERENCE.sub(replace, text)
            if new_text != text and not args.check:
                path.write_text(new_text, encoding="utf-8")
                print(f"updated {path.relative_to(REPO_ROOT)}")
    print(f"{n_references} equation references checked against {len(numbers)} labels.")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
