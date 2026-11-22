"""Closed physical surfaces and verified multipart 3MF assembly."""

from __future__ import annotations

from pathlib import Path

import lib3mf
from lib3mf import Lib3MF
import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import marching_cubes
import trimesh

from .geometry import Grid
from .labels import DISPLAY, ENVELOPE


def surface(mask: np.ndarray, grid: Grid) -> trimesh.Trimesh | None:
    if mask.shape != grid.shape:
        raise ValueError("A surface mask must match its physical grid.")
    bounds = find_objects(mask.astype(bool, copy=False), max_label=1)[0]
    if bounds is None:
        return None
    padded = np.pad(mask[bounds], 1)
    offset = np.array([axis.start - 1 for axis in bounds])
    # At exactly one half, diagonal voxel contacts can leave an open mesh.
    # Move the isosurface inward by 1e-5 of a voxel while preserving the binary mask.
    for level in (.5, .50001):
        vertices, faces, _, _ = marching_cubes(padded, level=level, allow_degenerate=False, method="lewiner")
        mesh = trimesh.Trimesh(grid.points(vertices + offset), faces, process=False)
        mesh.fix_normals(multibody=True)
        if mesh.is_volume:
            mesh.metadata["isovalue"] = level
            return mesh
    raise ValueError("A mask produced a surface that is not closed with outward normals.")


_PRINT_PRIORITY = ("lesion", "fibroglandular", "left_breast", "right_breast", ENVELOPE)


def printable_masks(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    occupied = np.zeros_like(next(iter(masks.values())), dtype=bool)
    parts = {}
    for label in _PRINT_PRIORITY:
        if label in masks:
            parts[label] = masks[label] & ~occupied
            occupied |= masks[label]
    return parts


def _write_assembly(path: Path, pieces: dict[str, trimesh.Trimesh]) -> None:
    wrapper = lib3mf.get_wrapper()
    model = wrapper.CreateModel()
    model.SetUnit(Lib3MF.ModelUnit.MilliMeter)
    assembly = model.AddComponentsObject()
    assembly.SetName("Breast MRI 3D research assembly")
    identity = wrapper.GetIdentityTransform()
    for name, mesh in pieces.items():
        vertices = [Lib3MF.Position(tuple(float(x) for x in row)) for row in mesh.vertices]
        triangles = [Lib3MF.Triangle(tuple(int(x) for x in row)) for row in mesh.faces]
        part = model.AddMeshObject()
        part.SetName(name)
        part.SetGeometry(vertices, triangles)
        assembly.AddComponent(part, identity)
    model.AddBuildItem(assembly, identity)
    model.QueryWriter("3mf").WriteToFile(str(path))


def _verify_assembly(path: Path, pieces: dict[str, trimesh.Trimesh], *, mesh_tolerance_mm=.002) -> None:
    wrapper = lib3mf.get_wrapper()
    model = wrapper.CreateModel()
    model.QueryReader("3mf").ReadFromFile(str(path))
    if model.GetUnit() != Lib3MF.ModelUnit.MilliMeter:
        raise ValueError("The assembly did not retain millimetre units.")
    build = model.GetBuildItems()
    if build.Count() != 1 or not build.MoveNext():
        raise ValueError("The assembly must open as one build item.")
    item = build.GetCurrent()
    identity = np.array(wrapper.GetIdentityTransform().Fields)
    if item.HasObjectTransform() and not np.allclose(np.array(item.GetObjectTransform().Fields), identity):
        raise ValueError("The 3MF build item has an unexpected transform.")
    assembly = model.GetComponentsObjectByID(item.GetObjectResourceID())
    if assembly.GetComponentCount() != len(pieces):
        raise ValueError("The 3MF assembly did not retain all expected parts.")
    seen = set()
    for i in range(assembly.GetComponentCount()):
        component = assembly.GetComponent(i)
        if component.HasTransform() and not np.allclose(np.array(component.GetTransform().Fields), identity):
            raise ValueError("A 3MF part has an unexpected placement transform.")
        obj = model.GetMeshObjectByID(component.GetObjectResourceID())
        name = obj.GetName()
        vertices = np.asarray([list(v.Coordinates) for v in obj.GetVertices()])
        if name not in pieces or name in seen or vertices.shape != pieces[name].vertices.shape or not np.allclose(
            vertices, pieces[name].vertices, atol=mesh_tolerance_mm, rtol=0
        ):
            raise ValueError(f"A 3MF part was moved or changed on reload: {name}")
        faces = np.asarray([list(f.Indices) for f in obj.GetTriangleIndices()])
        if not np.array_equal(faces, pieces[name].faces):
            raise ValueError(f"A 3MF part changed connectivity on reload: {name}")
        seen.add(name)


def export_meshes(root: Path, masks: dict[str, np.ndarray], grid: Grid) -> dict:
    scene = trimesh.Scene()
    # LPS millimetres -> right-handed, Y-up glTF metres.
    lps_to_glb = np.array([[-.001, 0, 0, 0], [0, 0, .001, 0], [0, .001, 0, 0], [0, 0, 0, 1]])
    anatomical = {}
    review_levels = {}
    for label, mask in masks.items():
        mesh = surface(mask, grid)
        if mesh is not None:
            anatomical[label] = float(mesh.area)
            review_levels[label] = mesh.metadata["isovalue"]
            mesh.apply_transform(lps_to_glb)
            mesh.visual.face_colors = DISPLAY[label][1]
            scene.add_geometry(mesh, node_name=label, geom_name=label)
    if len(scene.geometry):
        (root / "review.glb").write_bytes(scene.export(file_type="glb"))

    parts = {}
    for label, mask in printable_masks(masks).items():
        mesh = surface(mask, grid)
        if mesh is not None:
            parts[label] = mesh
    affine = np.eye(4)
    if parts:
        # Translate once for the entire assembly. Do not recenter individual parts.
        affine[:3, 3] = -np.min([part.bounds[0] for part in parts.values()], axis=0)
        folder = root / "print"
        folder.mkdir()
        for label, mesh in parts.items():
            mesh.apply_transform(affine)
            path = folder / f"{label}.stl"
            mesh.export(path)
            loaded = trimesh.load_mesh(path, process=False)
            if not np.allclose(loaded.bounds, mesh.bounds, rtol=0, atol=.002):
                raise ValueError(f"STL placement or scale changed for {label}.")
        assembly = folder / "assembly.3mf"
        _write_assembly(assembly, parts)
        _verify_assembly(assembly, parts)
    return {
        "surface_area_mm2": anatomical, "print_parts": sorted(parts),
        "lps_mm_to_print_mm": affine.tolist(), "lps_mm_to_glb_m": lps_to_glb.tolist(),
        "partition_priority": list(_PRINT_PRIORITY), "review_isovalues": review_levels,
        "print_isovalues": {label: mesh.metadata["isovalue"] for label, mesh in parts.items()},
        "assembly_3mf_verified": bool(parts),
    }
