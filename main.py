"""FluidBench entry point.

    python main.py list
    python main.py bench --backends all
    python main.py sweep
    python main.py sim --backend gpu --save vortex.gif

Run `python main.py --help` for the full set of options.
"""

import sys
from pathlib import Path

# Allow running this file directly from anywhere, not just from FluidBench/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fluidbench.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
