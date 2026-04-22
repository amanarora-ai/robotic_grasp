"""YCB object registry for Vega grasping environments.

Assets live in assets/ycb/. Each entry describes the object's
collision half-extents and mass so the env can reason about it
without parsing XML at runtime.
"""
import os
from typing import NamedTuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YCB_XML_DIR = os.path.join(BASE_DIR, "assets", "ycb", "xmls")


class YCBObject(NamedTuple):
    name: str           # key used in registry
    xml_path: str       # absolute path to standalone MJCF
    mass: float         # kg
    half_extents: tuple # (hx, hy, hz) metres — collision geom half-extents
    geom_type: str      # "box" | "cylinder" | "capsule"  (collision primitive)
    geom_z_offset: float = 0.0  # Z offset of collision geom centre from body origin


REGISTRY: dict[str, YCBObject] = {
    "potted_meat_can": YCBObject(
        name="potted_meat_can",
        xml_path=os.path.join(_YCB_XML_DIR, "potted_meat_can.xml"),
        mass=0.368,
        half_extents=(0.051, 0.030, 0.042),
        geom_type="box",
        geom_z_offset=-0.008,  # geom centre is 8mm below body origin
    ),
    "banana": YCBObject(
        name="banana",
        xml_path=os.path.join(_YCB_XML_DIR, "banana.xml"),
        mass=0.068,
        half_extents=(0.054, 0.089, 0.018),
        geom_type="capsule",
        geom_z_offset=0.0014,
    ),
    "mug": YCBObject(
        name="mug",
        xml_path=os.path.join(_YCB_XML_DIR, "mug.xml"),
        mass=0.102,
        half_extents=(0.046, 0.046, 0.040),
        geom_type="box",  # cylinder→box: MJX does not support cylinder-box collisions
        geom_z_offset=0.0016,
    ),
    "foam_brick": YCBObject(
        name="foam_brick",
        xml_path=os.path.join(_YCB_XML_DIR, "foam_brick.xml"),
        mass=0.030,
        half_extents=(0.026, 0.039, 0.026),
        geom_type="box",
        geom_z_offset=0.0083,
    ),
}

# Ordered list for random selection
OBJECT_NAMES = list(REGISTRY.keys())


def get(name: str) -> YCBObject:
    if name not in REGISTRY:
        raise KeyError(f"Unknown YCB object '{name}'. Available: {OBJECT_NAMES}")
    return REGISTRY[name]


def spawn_height(name: str, table_surface_z: float) -> float:
    """Z of body origin so the collision geom bottom sits 2mm above the table.

    body_z = table_z + 0.002 - geom_z_offset + half_extents_z
    (accounts for geom centre offset from body origin)
    """
    obj = get(name)
    return table_surface_z + 0.002 - obj.geom_z_offset + obj.half_extents[2]


def inject_into_scene(scene_xml_str: str, name: str) -> str:
    """Replace the cube <body name="object"> in a scene XML string with the
    named YCB object. Mesh file paths are made absolute so the result can be
    written to any directory and loaded with mujoco.MjModel.from_xml_path().
    """
    import copy
    import xml.etree.ElementTree as ET

    obj = get(name)
    ycb_xml_dir = os.path.dirname(obj.xml_path)

    obj_root = ET.parse(obj.xml_path).getroot()
    scene_root = ET.fromstring(scene_xml_str)
    wb = scene_root.find("worldbody")

    # Remove existing placeholder object body
    for body in list(wb.findall("body")):
        if body.get("name") == "object":
            wb.remove(body)
            break

    # Merge asset block; make mesh file paths absolute so temp-file loading works
    obj_asset = obj_root.find("asset")
    if obj_asset is not None:
        scene_asset = scene_root.find("asset")
        if scene_asset is None:
            scene_asset = ET.SubElement(scene_root, "asset")
        for child in obj_asset:
            child = copy.deepcopy(child)
            if child.tag == "mesh" and child.get("file"):
                child.set("file", os.path.abspath(
                    os.path.join(ycb_xml_dir, child.get("file"))))
            scene_asset.append(child)

    # Append YCB object body
    obj_wb = obj_root.find("worldbody")
    if obj_wb is not None:
        for body in obj_wb.findall("body"):
            wb.append(copy.deepcopy(body))

    return ET.tostring(scene_root, encoding="unicode")


def build_multi_object_scene_xml(scene_xml_str: str, object_names: list,
                                  include_visual_meshes: bool = False) -> str:
    """Inject multiple YCB objects into one scene XML.

    Each object gets an integer suffix on all body/joint/geom/site names
    (object_0, object_joint_0, object_geom_0, …) so they coexist without
    name collisions.  Objects are placed far off-table initially; the env's
    reset() positions the active object on the table each episode.

    Args:
        scene_xml_str: Base scene XML string (must contain <body name="object">
                       as a placeholder that will be removed).
        object_names:  Ordered list of YCB object names from REGISTRY.

    Returns:
        Modified XML string with all objects embedded.
    """
    import copy
    import xml.etree.ElementTree as ET

    scene_root = ET.fromstring(scene_xml_str)
    wb = scene_root.find("worldbody")

    # Remove placeholder cube body.
    for body in list(wb.findall("body")):
        if body.get("name") == "object":
            wb.remove(body)
            break

    scene_asset = scene_root.find("asset")
    if scene_asset is None:
        scene_asset = ET.SubElement(scene_root, "asset")

    for idx, name in enumerate(object_names):
        obj = get(name)
        ycb_xml_dir = os.path.dirname(obj.xml_path)
        obj_root = ET.parse(obj.xml_path).getroot()

        # Merge asset block (meshes only needed for visual rendering).
        if include_visual_meshes:
            obj_asset = obj_root.find("asset")
            if obj_asset is not None:
                for child in obj_asset:
                    child = copy.deepcopy(child)
                    if child.tag == "mesh" and child.get("file"):
                        child.set("file", os.path.abspath(
                            os.path.join(ycb_xml_dir, child.get("file"))))
                    scene_asset.append(child)

        # Inject body with all names suffixed by _{idx}.
        obj_wb = obj_root.find("worldbody")
        if obj_wb is None:
            continue
        for body in obj_wb.findall("body"):
            body = copy.deepcopy(body)
            body.set("name", f"object_{idx}")
            # Inactive objects start far off-table; reset() will reposition.
            body.set("pos", f"{20.0 + idx * 5.0} 0 0.05")  # above floor, minimal fall
            for jt in body.findall("freejoint"):
                jt.set("name", f"object_joint_{idx}")
            for geom in list(body.findall("geom")):
                gname = geom.get("name", "")
                if "visual" in gname:
                    if not include_visual_meshes:
                        body.remove(geom)  # strip mesh geom for training
                        continue
                    geom.set("name", f"object_visual_{idx}")
                else:
                    geom.set("name", f"object_geom_{idx}")
            for site in body.findall("site"):
                site.set("name", f"object_site_{idx}")
            wb.append(body)

    return ET.tostring(scene_root, encoding="unicode")


if __name__ == "__main__":
    print("YCB Object Registry")
    print(f"{'Name':<20} {'Mass':>8} {'Half-extents (m)':>30} {'Geom'}")
    print("-" * 70)
    for name, obj in REGISTRY.items():
        hx, hy, hz = obj.half_extents
        print(f"{name:<20} {obj.mass:>7.3f}kg  "
              f"({hx:.3f}, {hy:.3f}, {hz:.3f})  {obj.geom_type}")
        assert os.path.exists(obj.xml_path), f"Missing: {obj.xml_path}"
    print("\nAll XML paths verified.")
