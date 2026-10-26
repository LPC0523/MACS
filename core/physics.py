"""
core/physics.py  —  Biot-Savart + shimming optimiser
"""
import numpy as np
from scipy.optimize import lsq_linear
from scipy.io import loadmat, savemat
import nibabel as nib
from skimage.restoration import unwrap_phase


# ─────────────────────────────────────────────────────────────
#  Biot-Savart (vectorised NumPy)
# ─────────────────────────────────────────────────────────────

def biot_savart(coil_trace: np.ndarray,
                current: float,
                points: np.ndarray) -> np.ndarray:
    """
    Returns B [n_points, 3] (T) for a polygonal current loop.

    coil_trace : (N, 3)  coil nodes in metres
    current    : float   amperes
    points     : (M, 3)  observation points in metres
    """
    mu0 = 4 * np.pi * 1e-7
    centers = (coil_trace[:-1] + coil_trace[1:]) * 0.5   # (N-1, 3)
    dL      = coil_trace[1:]  - coil_trace[:-1]           # (N-1, 3)
    R       = points[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (M, N-1, 3)
    dist3   = (np.sum(R**2, axis=-1)**1.5) + 1e-30
    dB      = np.cross(dL, R) / dist3[..., np.newaxis]
    B       = mu0 * current * np.sum(dB, axis=1) / (4 * np.pi)
    return B                                               # (M, 3)


# ─────────────────────────────────────────────────────────────
#  Field-basis matrix  A[m, n] = Bz_n(point_m) per 1 A
# ─────────────────────────────────────────────────────────────

def compute_field_basis(coil_list, obs_points, progress_cb=None):
    """
    Returns A (M, N_coils) in µT/A  [i.e. unit-current Bz contributions].
    progress_cb: optional callable(i, total)
    """
    M = obs_points.shape[0]
    N = len(coil_list)
    A = np.zeros((M, N), dtype=np.float32)
    for i, coil in enumerate(coil_list):
        B      = biot_savart(coil, 1.0, obs_points)
        A[:, i] = B[:, 2]          # Bz component
        if progress_cb:
            progress_cb(i + 1, N)
    return A * 1e6   # convert T → µT for numerical stability


# ─────────────────────────────────────────────────────────────
#  Linear shim optimiser
# ─────────────────────────────────────────────────────────────

def shim_optimize(A, b0_flat, imax=3.0, weights=None):
    """
    Solve  min || A·x + b0 ||²   s.t.  |x_i| ≤ imax

    A        : (M, N)  field basis in same units as b0_flat
    b0_flat  : (M,)    measured B0 in ppm
    imax     : float   max current per channel (A)
    weights  : (M,) optional spatial weights (e.g. brain-centre mask)

    Returns
        currents     : (N,)
        b0_corrected : (M,)
        metrics      : dict
    """
    if weights is not None:
        w = np.sqrt(weights)
        A_w  = A   * w[:, None]
        b0_w = b0_flat * w
    else:
        A_w, b0_w = A, b0_flat

    res = lsq_linear(A_w, -b0_w, bounds=(-imax, imax), method='bvls')
    currents     = res.x
    b0_corrected = A @ currents + b0_flat

    metrics = {
        "std_before"  : float(np.std(b0_flat)),
        "std_after"   : float(np.std(b0_corrected)),
        "rmse_before" : float(np.sqrt(np.mean(b0_flat**2))),
        "rmse_after"  : float(np.sqrt(np.mean(b0_corrected**2))),
        "improvement" : float(1 - np.std(b0_corrected) / np.std(b0_flat)),
    }
    return currents, b0_corrected, metrics


# ─────────────────────────────────────────────────────────────
#  I/O helpers
# ─────────────────────────────────────────────────────────────

def load_b0_mat(file_obj):
    """Load B0 map from .mat file.  Returns coords (N,3), b0 (N,) in ppm."""
    data = loadmat(file_obj)
    X = data["X"].flatten()
    Y = data["Y"].flatten()
    Z = data["Z"].flatten()
    b0 = data["B0_data"].flatten()
    coords = np.stack([X, Y, Z], axis=1)
    return coords, b0


def load_b0_nifti(file_obj):
    """
    Load a B0 map from a .nii / .nii.gz file.
    Accepts a file-like object (st.file_uploader) or a path string.

    Returns
    -------
    coords : (N, 3)  world coordinates in metres (non-zero voxels only)
    b0     : (N,)    field values in the unit stored in the file
    """
    import tempfile, os as _os
    tmp_path = None
    try:
        if hasattr(file_obj, "read"):
            suffix = ".nii.gz" if getattr(file_obj, "name", "").endswith(".gz") else ".nii"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_obj.read())
                tmp_path = tmp.name
            img = nib.load(tmp_path)
        else:
            img = nib.load(str(file_obj))

        # get_fdata() must be called BEFORE unlinking the temp file —
        # nibabel reads .gz lazily and still needs the file on disk here.
        data   = img.get_fdata(dtype=np.float32)
        affine = img.affine.copy()
    finally:
        if tmp_path and _os.path.exists(tmp_path):
            _os.unlink(tmp_path)

    shape  = data.shape[:3]

    ii, jj, kk = np.meshgrid(np.arange(shape[0]),
                              np.arange(shape[1]),
                              np.arange(shape[2]), indexing="ij")
    vox   = np.stack([ii.ravel(), jj.ravel(), kk.ravel(),
                      np.ones(ii.size)], axis=0)        # (4, N)
    world = (affine @ vox)[:3].T                        # (N, 3) mm

    b0_flat = data.ravel()
    mask    = b0_flat != 0
    coords  = world[mask] / 1000.0                      # mm -> metres

    # Also return the full 3D volume + affine for slice-based viewers
    # Callers can ignore these extra returns for backward compat
    return coords, b0_flat[mask], data, affine


def load_coils_mat(file_obj):
    """Load coil list from .mat.  Returns list of (Ni, 3) arrays."""
    data  = loadmat(file_obj)
    keys  = sorted([k for k in data.keys() if k.startswith("coil")])
    coils = []
    for k in keys:
        s = data[k]
        # handle both plain array and struct
        try:
            X = s["X"][0, 0].flatten()
            Y = s["Y"][0, 0].flatten()
            Z = s["Z"][0, 0].flatten()
            coils.append(np.vstack([X, Y, Z]).T)
        except (TypeError, IndexError):
            arr = np.array(s, dtype=float)
            if arr.shape[0] == 3:
                arr = arr.T
            coils.append(arr)
    return coils, keys


def load_coils_obj(file_obj):
    """Parse Wavefront .obj → list of (Ni, 3) coil arrays."""
    lines    = file_obj.read().decode("utf-8").splitlines()
    vertices = []
    loops    = []
    for ln in lines:
        if ln.startswith("v "):
            vertices.append(list(map(float, ln.split()[1:4])))
        elif ln.startswith("l "):
            idx = [int(p) - 1 for p in ln.split()[1:]]
            if idx:
                loops.append(idx)
    verts = np.array(vertices)
    coils = [verts[np.array(idx)] for idx in loops]
    return coils


def load_coils_txt(file_obj):
    """Parse multi-coil txt (blocks separated by 'New strike' or blank lines)."""
    text   = file_obj.read().decode("utf-8")
    blocks = [b.strip() for b in text.replace("New strike", "\n\n").split("\n\n") if b.strip()]
    coils  = []
    for block in blocks:
        rows = []
        for ln in block.splitlines():
            parts = ln.strip().split()
            if len(parts) >= 3:
                rows.append(list(map(float, parts[:3])))
        if rows:
            coils.append(np.array(rows))
    return coils
