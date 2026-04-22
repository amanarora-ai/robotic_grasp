"""Convert Vega URDF to a clean MJCF for MuJoCo Playground training.

Produces xmls/vega_right_arm.xml with:
  - Left side + torso + head joints locked via high stiffness/damping (not removed)
  - Left arm resting pose: arms hanging down (j2 springref tuned per side)
  - Position actuators for right arm (7) and right hand (11)
  - Sites on right palm and fingertips for reward computation
"""
import os
import xml.etree.ElementTree as ET
import mujoco

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(
    BASE_DIR,
    "dexmate-urdf/robots/humanoid/vega_1u/vega_1u_f5d6_mujoco.urdf",
)
TMP_XML = "/tmp/vega_raw.xml"
OUT_ROBOT_XML = os.path.join(BASE_DIR, "xmls/vega_right_arm.xml")

# Locked joints: name -> (springref, stiffness, damping)
# L_arm_j2 springref=+1.3 and L_arm_j4=-0.5 puts the left arm hanging down naturally.
LOCK_CONFIG = {
    "Lift":       (0.0,  1000000, 30000),
    "torso_flip": (0.0,  10000, 1000),
    "head_j1":    (0.0,   5000,  500),
    "head_j2":    (0.0,   5000,  500),
    "head_j3":    (0.0,   5000,  500),
    "L_arm_j1":   (1.57,  2000,  100),  # arm hanging down
    "L_arm_j2":   (0.0,   2000,  100),
    "L_arm_j3":   (0.0,   2000,  100),
    "L_arm_j4":   (0.0,   2000,  100),
    "L_arm_j5":   (0.0,   2000,  100),
    "L_arm_j6":   (0.0,   2000,  100),
    "L_arm_j7":   (0.0,   2000,  100),
    "L_th_j0":    (0.0,    500,   20),
    "L_th_j1":    (0.0,    500,   20),
    "L_th_j2":    (0.0,    500,   20),
    "L_ff_j1":    (0.0,    500,   20),
    "L_ff_j2":    (0.0,    500,   20),
    "L_mf_j1":    (0.0,    500,   20),
    "L_mf_j2":    (0.0,    500,   20),
    "L_rf_j1":    (0.0,    500,   20),
    "L_rf_j2":    (0.0,    500,   20),
    "L_lf_j1":    (0.0,    500,   20),
    "L_lf_j2":    (0.0,    500,   20),
}

RIGHT_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
RIGHT_HAND_JOINTS = [
    "R_th_j0", "R_th_j1", "R_th_j2",
    "R_ff_j1", "R_ff_j2",
    "R_mf_j1", "R_mf_j2",
    "R_rf_j1", "R_rf_j2",
    "R_lf_j1", "R_lf_j2",
]

PALM_CANDIDATES = ["R_arm_l7", "R_hand_base", "right_hand_base", "R_wrist_link"]
FINGERTIP_CANDIDATES = {
    "right_thumb_tip":  (["R_th_l2", "r_th_l2"],  "0.0211  0.0010 -0.0043"),
    "right_index_tip":  (["R_ff_l2", "r_ff_l2"],  "-0.020  0.0    -0.038"),
    "right_middle_tip": (["R_mf_l2", "r_mf_l2"],  "-0.020  0.0    -0.038"),
    "right_ring_tip":   (["R_rf_l2", "r_rf_l2"],  "-0.018  0.0    -0.040"),
    "right_little_tip": (["R_lf_l2", "r_lf_l2"],  "-0.013  0.0    -0.028"),
}


def find_body(worldbody, name):
    if worldbody.get("name") == name:
        return worldbody
    for body in worldbody.iter("body"):
        if body.get("name") == name:
            return body
    return None


def lock_joint(worldbody, joint_name, springref, stiffness, damping):
    for body in worldbody.iter("body"):
        for j in body.findall("joint"):
            if j.get("name") == joint_name:
                j.set("stiffness", str(stiffness))
                j.set("damping", str(damping))
                j.set("springref", str(springref))
                return True
    return False


def main():
    m = mujoco.MjModel.from_xml_path(URDF_PATH)
    mujoco.mj_saveLastXML(TMP_XML, m)
    print(f"Loaded: {m.njnt} joints, {m.nbody} bodies")

    tree = ET.parse(TMP_XML)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    # Use relative meshdir so the XML works on any machine
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", "../dexmate-urdf/robots/humanoid/vega_1u")
    compiler.set("fitaabb", "true")

    # Rotate robot 90° CCW from top so right arm faces +x (toward table)
    root_body = find_body(worldbody, "lift_link")
    if root_body is not None:
        root_body.set("euler", "0 0 1.5708")

    # Lock passive joints with high stiffness (preserves joint so springref works)
    for jname, (springref, stiffness, damping) in LOCK_CONFIG.items():
        if not lock_joint(worldbody, jname, springref, stiffness, damping):
            print(f"WARNING: joint not found: {jname}")

    # Add position actuators for active right-side joints
    actuator = ET.SubElement(root, "actuator")
    for jname in RIGHT_ARM_JOINTS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        lo, hi = m.jnt_range[jid]
        ET.SubElement(actuator, "position", {
            "name": f"act_{jname}",
            "joint": jname,
            "kp": "100",
            "kv": "20",
            "ctrlrange": f"{lo:.4f} {hi:.4f}",
            "forcerange": "-200 200",
        })
    for jname in RIGHT_HAND_JOINTS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        lo, hi = m.jnt_range[jid]
        ET.SubElement(actuator, "position", {
            "name": f"act_{jname}",
            "joint": jname,
            "kp": "10",
            "kv": "2",
            "ctrlrange": f"{lo:.4f} {hi:.4f}",
            "forcerange": "-20 20",
        })

    # Palm site
    for cand in PALM_CANDIDATES:
        body = find_body(worldbody, cand)
        if body is not None:
            ET.SubElement(body, "site", {
                "name": "right_palm_site",
                "pos": "0.194 0.011 -0.031",
                "size": "0.02",
                "rgba": "1 0 0 0.3",
            })
            print(f"Added right_palm_site → '{cand}'")
            break

    # Fingertip sites
    for site_name, (candidates, pos) in FINGERTIP_CANDIDATES.items():
        for cand in candidates:
            body = find_body(worldbody, cand)
            if body is not None:
                ET.SubElement(body, "site", {
                    "name": site_name,
                    "pos": pos,
                    "size": "0.008",
                    "rgba": "0 1 0 0.3",
                })
                print(f"Added {site_name} → '{cand}'")
                break

    # Classify active bodies
    active_joint_names = set(RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS)
    arm_bodies, finger_bodies = set(), set()
    for body in worldbody.iter("body"):
        for j in body.findall("joint"):
            jname = j.get("name", "")
            if jname in RIGHT_ARM_JOINTS:
                arm_bodies.add(body.get("name"))
            elif jname in RIGHT_HAND_JOINTS:
                finger_bodies.add(body.get("name"))

    active_bodies = arm_bodies | finger_bodies
    # Include children of arm and finger bodies (e.g. R_hand_base, terminal finger links)
    for body in worldbody.iter("body"):
        if body.get("name") in (arm_bodies | finger_bodies):
            for child in body.findall("body"):
                active_bodies.add(child.get("name"))

    fingertip_bodies = set()
    for _, (candidates, _) in FINGERTIP_CANDIDATES.items():
        for cand in candidates:
            if find_body(worldbody, cand) is not None:
                fingertip_bodies.add(cand)
                break

    _DUMMY_GEOM = {"type": "sphere", "size": "0.001", "mass": "0.001",
                   "rgba": "0 0 0 0", "contype": "0", "conaffinity": "0"}
    _TIP_GEOM   = {"type": "sphere", "size": "0.012",
                   "rgba": "0 1 0 0.3", "contype": "1", "conaffinity": "1"}

    _PALM_MESHES = {"R_hand_base"}
    _FITSCALE_OVERRIDE = {"R_th_l1": "0.8"}  # thumb middle link oversized

    def fitted_geom(mesh_name, geom_type, rgba, orig_attrs=None, fitscale="1.0"):
        d = {"type": geom_type, "mesh": mesh_name, "fitscale": fitscale,
             "rgba": rgba, "group": "0", "contype": "1", "conaffinity": "1"}
        for k in ("pos", "quat"):
            if orig_attrs and k in orig_attrs:
                d[k] = orig_attrs[k]
        return d

    active_mesh_names = set()
    n_replaced = 0
    for body in worldbody.iter("body"):
        geoms = body.findall("geom")
        if not geoms:
            continue
        bname = body.get("name", "")
        mesh_geoms = [g for g in geoms if g.get("type") == "mesh"]

        for g in geoms:
            body.remove(g)
        n_replaced += len(geoms)

        if bname in fingertip_bodies:
            # Capsule only — site already marks the tip; sphere at joint origin was misleading
            for mg in mesh_geoms:
                mname = mg.get("mesh", "")
                scale = _FITSCALE_OVERRIDE.get(bname, "1.0")
                ET.SubElement(body, "geom", fitted_geom(mname, "capsule", "0.5 0.5 0.5 1", mg.attrib, scale))
                active_mesh_names.add(mname)
        elif bname in active_bodies and mesh_geoms:
            for mg in mesh_geoms:
                mname = mg.get("mesh", "")
                scale = _FITSCALE_OVERRIDE.get(bname, "1.0")
                if mname in _PALM_MESHES:
                    ET.SubElement(body, "geom", fitted_geom(mname, "box", "0.6 0.6 0.6 1", mg.attrib, "0.75"))
                elif bname in finger_bodies or bname in active_bodies:
                    rgba = "0.7 0.7 0.7 1" if bname in arm_bodies else "0.5 0.5 0.5 1"
                    ET.SubElement(body, "geom", fitted_geom(mname, "capsule", rgba, mg.attrib, scale))
                active_mesh_names.add(mname)
        else:
            ET.SubElement(body, "geom", _DUMMY_GEOM)

    # Remove worldbody-level mesh geom (artifact from mj_saveLastXML)
    for g in worldbody.findall("geom"):
        worldbody.remove(g)

    # Remove mesh assets for inactive bodies; keep active ones for fitaabb
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            if mesh.get("name") not in active_mesh_names:
                asset.remove(mesh)

    print(f"Replaced {n_replaced} geoms; fitted primitives for {len(active_mesh_names)} active meshes")
    print(f"  arm capsules: {arm_bodies}")
    print(f"  fingertip spheres: {fingertip_bodies}")

    tree.write(OUT_ROBOT_XML, xml_declaration=True, encoding="unicode")
    print(f"\nSaved: {OUT_ROBOT_XML}")
    print(f"Actuators: {len(RIGHT_ARM_JOINTS)} arm + {len(RIGHT_HAND_JOINTS)} hand = "
          f"{len(RIGHT_ARM_JOINTS) + len(RIGHT_HAND_JOINTS)} total")


if __name__ == "__main__":
    main()
