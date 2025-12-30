import warnings

import numpy as np

from pylops.utils.typing import NumericNDArray


def ensure_iava_is_unique(
    iava: NumericNDArray,
    axis: int | None = None,
) -> None:
    _, count = np.unique(
        iava,
        axis=axis,
        return_counts=True,
    )

    if np.any(count > 1):
        raise ValueError("Found repeated/non-unique values in iava.")

    return


def clip_iava_above_last_sample_index(
    iava: NumericNDArray,
    sample_size: int,
) -> None:
    # ensure that samples are not beyond the last sample, in that case set to
    # penultimate sample and raise a warning
    last_sample_index = sample_size - 1
    outside = iava >= last_sample_index
    if np.any(outside):
        warnings.warn(
            f"At least one value in iava is beyond the penultimate sample index "
            f"{last_sample_index}. Out-of-bound-values are forced below penultimate "
            f"sample."
        )

        # NOTE: ``numpy.nextafter(x, -np.inf)`` gives the closest float-value that is
        #       less than ``x``, i.e., this logic clips ``iava`` to the highest possible
        #       value that is still below the last sample
        iava[np.where(outside)] = np.nextafter(last_sample_index, -np.inf)

    ensure_iava_is_unique(iava=iava)

    return
