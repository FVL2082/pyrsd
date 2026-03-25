"""
pyrsd/utils/export.py
Converts processed arrays to external formats for visualisation and storage.
"""

import base64
import struct
import numpy as np
from pathlib import Path

_VTK_TYPES = {
    np.dtype("float32"): "Float32",
    np.dtype("float64"): "Float64",
    np.dtype("int32"): "Int32",
    np.dtype("uint32"): "UInt32",
    np.dtype("uint8"): "UInt8",
}


def _encode(arr: np.ndarray) -> str:
    """Base64 encode array with leading UInt32 byte count."""
    arr = np.ascontiguousarray(arr)
    raw = arr.tobytes()
    header = struct.pack("<I", len(raw))
    return base64.b64encode(header + raw).decode("ascii")


def _data_array_xml(name: str, arr: np.ndarray, ncomp: int = 1) -> list[str]:
    """Return XML block for a VTK DataArray."""
    dtype = arr.dtype
    vtype = _VTK_TYPES.get(dtype)

    if vtype is None:
        raise ValueError(f"Unsupported dtype for VTK: {dtype}")

    encoded = _encode(arr)

    return [
        f'  <DataArray type="{vtype}" Name="{name}" '
        f'NumberOfComponents="{ncomp}" format="binary">',
        f'    {encoded}',
        "  </DataArray>",
    ]


def _write_vtk(
    field: np.ndarray,
    output_path: str,
    spacing: tuple[float, ...],
    origin: tuple[float, ...],
    field_name: str = "field",
) -> None:
    """Write a single .vti file."""
    
    field = np.ascontiguousarray(field)

    is_3d = len(spacing) == 3
    n_spatial = 3 if is_3d else 2
    ncomp = field.shape[n_spatial] if field.ndim > n_spatial else 1

    h, w = field.shape[0], field.shape[1]
    d = field.shape[2] if is_3d else 1

    ox, oy, oz = (*origin, 0.0)[:3]
    dx, dy, dz = (*spacing, 1.0)[:3]

    ext = f"0 {w-1} 0 {h-1} 0 {d-1}"

    if ncomp > 1:
        flat = field.reshape(-1, ncomp).astype(np.float64).ravel()
        spatial_flat = field.reshape(-1, ncomp)
        mask = ~np.isnan(spatial_flat).any(axis=1)
    else:
        flat = field.ravel().astype(np.float64)
        spatial_flat = field.ravel()
        mask = ~np.isnan(spatial_flat)

    mask = mask.astype(np.uint8)

    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32">',
        f'  <ImageData WholeExtent="{ext}" Origin="{ox} {oy} {oz}" Spacing="{dx} {dy} {dz}">',
        f'    <Piece Extent="{ext}">',
        "      <PointData>",
        *_data_array_xml(field_name, flat, ncomp),
        *_data_array_xml("vtkValidPointMask", mask, 1),
        "      </PointData>",
        "    </Piece>",
        "  </ImageData>",
        "</VTKFile>",
    ]

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _write_pvd(output_path: str, vti_files: list[str]) -> None:
    """Write ParaView time-series collection (.pvd)."""
    
    entries = [
        f'  <DataSet timestep="{t}" part="0" file="{Path(f).name}"/>'
        for t, f in enumerate(vti_files)
    ]

    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
        *entries,
        "  </Collection>",
        "</VTKFile>",
    ]

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def save_as_vtk(
    stack: np.ndarray,
    output: str,
    spacing: tuple = (1.0, 1.0),
    origin: tuple = (0.0, 0.0),
    field_name: str = "field",
    time: bool = False,
) -> None:
    """saves array or time stack as VTK ImageData (.vti) and optional .pvd series"""
    output = str(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    if not time:
        out_path = output + ".vti"
        _write_vtk(stack, out_path, spacing, origin, field_name)
        print(f"Written: {out_path}")
        return

    nt = stack.shape[0]
    w = len(str(nt))

    vti_files = []
    for t in range(nt):
        path = f"{output}_{t:0{w}d}.vti"
        _write_vtk(stack[t], path, spacing, origin, field_name)
        vti_files.append(path)

    pvd_path = output + ".pvd"
    _write_pvd(pvd_path, vti_files)

    print(f"Written: {nt} frames + {pvd_path}")


def save_hdf5(data: dict[str, np.ndarray], path: str) -> None:
    """Save multiple arrays into one HDF5 file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)  
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 export.\n"
            "Install with: pip install h5py"
        ) from e

    with h5py.File(path, "w") as f:
        for name, arr in data.items():
            f.create_dataset(name, data=arr)

def save_csv_profile(x: np.ndarray, y: np.ndarray, path: str, x_label: str = "position", y_label: str = "value") -> None:
    """saves a 1D profile as CSV"""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = f"{x_label},{y_label}"
    np.savetxt(str(dest), np.column_stack([x, y]), delimiter=",", header=header, comments="")
