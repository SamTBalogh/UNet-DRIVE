from __future__ import annotations

import numpy as np

from metrics import dice_score_numpy


def main() -> None:
    mask = np.array([[1, 0], [0, 1]], dtype=np.float32)
    same = np.array([[1, 0], [0, 1]], dtype=np.float32)
    different = np.array([[0, 1], [1, 0]], dtype=np.float32)
    partial = np.array([[1, 0], [1, 0]], dtype=np.float32)
    empty = np.zeros((2, 2), dtype=np.float32)

    cases = {
        "same": dice_score_numpy(mask, same),
        "different": dice_score_numpy(mask, different),
        "partial": dice_score_numpy(mask, partial),
        "empty_vs_empty": dice_score_numpy(empty, empty),
    }

    for name, value in cases.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
