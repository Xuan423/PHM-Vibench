import scipy.io as sio
import numpy as np


def _extract_nested(obj, path=("Signal", "y_values", "values")):
    """Traverse ``path`` keys/attributes inside a MATLAB struct-like object."""
    cur = obj
    for p in path:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            # MATLAB structs from ``loadmat`` expose fields as attributes
            cur = getattr(cur, p, None)
        if cur is None:
            return None
    return cur


def read(file_path, *_args, path=("Signal", "y_values", "values"), **_kwargs):
    """Load SDUST ``.mat`` file and return signal samples as ``(length, channels)`` array."""
    mat_data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    values = _extract_nested(mat_data, path=path)
    if values is None:
        raise KeyError(
            f"Unable to locate {'/'.join(path)} in '{file_path}'."
        )

    array = np.array(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    elif array.ndim > 2:
        raise ValueError(
            f"Expect 1D or 2D signal array, got shape {array.shape} from '{file_path}'."
        )
    return array


if __name__ == "__main__":
    mat_path = "/mnt/e/dataset/PHMbench-raw_data/raw/RM_032_SDUST/IF0.2/IF0.2 800~1500 0.mat"
    data = read(mat_path)

    # Attempt to find sampling frequency if present
    mat_raw = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    fs_candidates = []
    for k, v in mat_raw.items():
        if isinstance(v, (int, float, np.integer, np.floating)) and (
            "fs" in k.lower() or "freq" in k.lower() or "sample" in k.lower()
        ):
            fs_candidates.append((k, v))

    print("Data shape:", data.shape)
