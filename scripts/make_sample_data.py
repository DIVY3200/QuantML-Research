"""Write the clearly labelled SYNTHETIC sample dataset used for offline demos and CSV testing.

    python scripts/make_sample_data.py

The output is generated noise (GARCH-style volatility, no built-in predictability). It is not
market data and must never be presented as real performance.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantml.data_loader import generate_synthetic_ohlcv  # noqa: E402

OUTPUT = ROOT / "data" / "sample" / "SYNTHETIC_sample_ohlcv.csv"


def main() -> None:
    frame = generate_synthetic_ohlcv(n_days=2500, seed=42)
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.round(4).to_csv(OUTPUT, index=False)
    print(f"Wrote {len(frame)} SYNTHETIC rows to {OUTPUT}")


if __name__ == "__main__":
    main()
