from __future__ import annotations

import numpy as np


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 0,
    fov_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Remove connected vessel components smaller than ``min_size`` pixels."""

    mask_bool = np.asarray(mask).astype(bool)
    if fov_mask is not None:
        mask_bool &= np.asarray(fov_mask).astype(bool)
    if min_size <= 1:
        return mask_bool.astype(np.uint8)

    kept = np.zeros_like(mask_bool, dtype=bool)
    visited = np.zeros_like(mask_bool, dtype=bool)
    height, width = mask_bool.shape
    rows, cols = np.nonzero(mask_bool)

    for start_row, start_col in zip(rows.tolist(), cols.tolist()):
        if visited[start_row, start_col]:
            continue
        component = collect_component(
            mask=mask_bool,
            visited=visited,
            start_row=start_row,
            start_col=start_col,
            height=height,
            width=width,
        )
        if len(component) >= min_size:
            component_rows, component_cols = zip(*component)
            kept[component_rows, component_cols] = True
    return kept.astype(np.uint8)


def collect_component(
    mask: np.ndarray,
    visited: np.ndarray,
    start_row: int,
    start_col: int,
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    stack = [(start_row, start_col)]
    visited[start_row, start_col] = True
    component: list[tuple[int, int]] = []

    while stack:
        row, col = stack.pop()
        component.append((row, col))
        for row_delta in (-1, 0, 1):
            for col_delta in (-1, 0, 1):
                if row_delta == 0 and col_delta == 0:
                    continue
                next_row = row + row_delta
                next_col = col + col_delta
                if not (0 <= next_row < height and 0 <= next_col < width):
                    continue
                if visited[next_row, next_col] or not mask[next_row, next_col]:
                    continue
                visited[next_row, next_col] = True
                stack.append((next_row, next_col))
    return component
