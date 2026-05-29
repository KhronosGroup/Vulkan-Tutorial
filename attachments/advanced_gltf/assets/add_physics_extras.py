#!/usr/bin/env python3
# Copyright (c) 2026 Holochip Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 the "License";
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
add_physics_extras.py  —  Annotates glTF skeleton nodes with the physics extras
schema used by the Advanced glTF tutorial (ColliderDef / ConstraintDef).

Usage:
    python add_physics_extras.py input.gltf output.gltf [--config ragdoll.json]

If --config is omitted, the script uses built-in heuristics that recognise
common bone-naming conventions (Mixamo, Blender Rigify, generic "bone_*").

Output extras schema (matches ColliderDef / ConstraintDef in node.h):
    node.extras.collider  = { "shape", "radius", "half_height",
                               "box_half_extents", "mass",
                               "collision_group", "collision_mask" }
    node.extras.constraint = { "type", "swing_limit_deg", "twist_limit_deg",
                                "hinge_min_deg", "hinge_max_deg",
                                "hinge_axis", "parent_bone" }

NOTE: Physics extras are a tutorial-specific extension, not a registered glTF
extension.  They will pass the Khronos glTF-Validator only if the validator is
run without the --strict-extra flag.
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bone heuristic database
# Each entry is (pattern_substrings, collider, constraint).
# Patterns are checked case-insensitively; first match wins.
# ---------------------------------------------------------------------------
BONE_RULES = [
    # HEAD / NECK
    (["head"],
     {"shape": "CAPSULE", "radius": 0.10, "half_height": 0.06,
      "mass": 4.0, "collision_group": "character", "collision_mask": "world"},
     {"type": "BALL_SOCKET", "swing_limit_deg": 50.0, "twist_limit_deg": 30.0}),

    (["neck"],
     {"shape": "CAPSULE", "radius": 0.06, "half_height": 0.06,
      "mass": 2.0, "collision_group": "character", "collision_mask": "world"},
     {"type": "BALL_SOCKET", "swing_limit_deg": 40.0, "twist_limit_deg": 25.0}),

    # SPINE / PELVIS / HIPS
    (["pelvis", "hips", "root"],
     {"shape": "BOX", "box_half_extents": [0.14, 0.08, 0.10],
      "mass": 8.0, "collision_group": "character", "collision_mask": "world"},
     {"type": "NONE"}),

    (["spine", "chest", "torso"],
     {"shape": "BOX", "box_half_extents": [0.12, 0.10, 0.08],
      "mass": 6.0, "collision_group": "character", "collision_mask": "world"},
     {"type": "BALL_SOCKET", "swing_limit_deg": 20.0, "twist_limit_deg": 15.0}),

    # UPPER LIMBS
    (["upperarm", "upper_arm", "arm_upper", "uparm"],
     {"shape": "CAPSULE", "radius": 0.05, "half_height": 0.14,
      "mass": 2.5, "collision_group": "character", "collision_mask": "world"},
     {"type": "BALL_SOCKET", "swing_limit_deg": 80.0, "twist_limit_deg": 60.0}),

    (["lowerarm", "lower_arm", "arm_lower", "forearm", "loarm"],
     {"shape": "CAPSULE", "radius": 0.04, "half_height": 0.13,
      "mass": 1.5, "collision_group": "character", "collision_mask": "world"},
     {"type": "HINGE",
      "hinge_axis": [0.0, 0.0, 1.0],
      "hinge_min_deg": -140.0, "hinge_max_deg": 0.0}),

    (["hand", "wrist"],
     {"shape": "BOX", "box_half_extents": [0.04, 0.03, 0.07],
      "mass": 0.5, "collision_group": "character", "collision_mask": "world"},
     {"type": "BALL_SOCKET", "swing_limit_deg": 60.0, "twist_limit_deg": 30.0}),

    # LOWER LIMBS
    (["upperleg", "upper_leg", "leg_upper", "thigh", "upleg"],
     {"shape": "CAPSULE", "radius": 0.07, "half_height": 0.20,
      "mass": 5.0, "collision_group": "character", "collision_mask": "world"},
     {"type": "BALL_SOCKET", "swing_limit_deg": 70.0, "twist_limit_deg": 30.0}),

    (["lowerleg", "lower_leg", "leg_lower", "shin", "calf", "loleg"],
     {"shape": "CAPSULE", "radius": 0.05, "half_height": 0.18,
      "mass": 3.0, "collision_group": "character", "collision_mask": "world"},
     {"type": "HINGE",
      "hinge_axis": [1.0, 0.0, 0.0],
      "hinge_min_deg": 0.0, "hinge_max_deg": 140.0}),

    (["foot", "ankle"],
     {"shape": "BOX", "box_half_extents": [0.05, 0.03, 0.10],
      "mass": 1.0, "collision_group": "character", "collision_mask": "world"},
     {"type": "HINGE",
      "hinge_axis": [1.0, 0.0, 0.0],
      "hinge_min_deg": -30.0, "hinge_max_deg": 45.0}),

    (["toe"],
     {"shape": "CAPSULE", "radius": 0.02, "half_height": 0.02,
      "mass": 0.2, "collision_group": "character", "collision_mask": "world"},
     {"type": "HINGE",
      "hinge_axis": [1.0, 0.0, 0.0],
      "hinge_min_deg": -30.0, "hinge_max_deg": 30.0}),
]

# Fallback for joints whose name matches none of the rules above.
DEFAULT_COLLIDER = {
    "shape": "CAPSULE", "radius": 0.04, "half_height": 0.06,
    "mass": 1.0, "collision_group": "character", "collision_mask": "world",
}
DEFAULT_CONSTRAINT = {
    "type": "BALL_SOCKET", "swing_limit_deg": 45.0, "twist_limit_deg": 30.0,
}


def match_bone(name: str):
    """Return (collider, constraint) dicts for the given bone name."""
    lower = name.lower()
    # Strip common left/right prefixes/suffixes: l_, r_, _l, _r, left_, right_, .L, .R
    for prefix in ("left_", "right_", "l_", "r_"):
        if lower.startswith(prefix):
            lower = lower[len(prefix):]
    for suffix in ("_left", "_right", "_l", "_r", ".l", ".r"):
        if lower.endswith(suffix):
            lower = lower[: -len(suffix)]

    for patterns, collider, constraint in BONE_RULES:
        if any(p in lower for p in patterns):
            return dict(collider), dict(constraint)
    return dict(DEFAULT_COLLIDER), dict(DEFAULT_CONSTRAINT)


def find_parent_bone_name(gltf: dict, node_idx: int) -> str:
    """Return the name of node_idx's parent if it is also a joint, else ''."""
    for idx, node in enumerate(gltf.get("nodes", [])):
        if node_idx in node.get("children", []):
            return node.get("name", "")
    return ""


def collect_joint_indices(gltf: dict) -> set:
    """Return the set of node indices that are referenced by any skin."""
    joints: set = set()
    for skin in gltf.get("skins", []):
        joints.update(skin.get("joints", []))
    return joints


def annotate(gltf: dict, config: dict | None, dry_run: bool) -> tuple[int, int]:
    """
    Add physics extras to skeleton nodes.
    Returns (nodes_annotated, nodes_skipped).
    """
    joint_indices = collect_joint_indices(gltf)
    if not joint_indices:
        print("  WARNING: No skins found in file — no joints to annotate.", file=sys.stderr)
        return 0, 0

    nodes = gltf.setdefault("nodes", [])
    annotated = 0
    skipped = 0

    for idx in sorted(joint_indices):
        if idx >= len(nodes):
            skipped += 1
            continue
        node = nodes[idx]
        name = node.get("name", f"node_{idx}")

        # Look up in explicit config first, then heuristics.
        if config and name in config:
            collider   = config[name].get("collider",   dict(DEFAULT_COLLIDER))
            constraint = config[name].get("constraint", dict(DEFAULT_CONSTRAINT))
        else:
            collider, constraint = match_bone(name)

        # Attach parent bone name to constraint for context.
        parent_name = find_parent_bone_name(gltf, idx)
        if parent_name:
            constraint["parent_bone"] = parent_name

        if not dry_run:
            extras = node.setdefault("extras", {})
            extras["collider"]   = collider
            extras["constraint"] = constraint

        print(f"  [{idx:3d}] {name:<30s}  shape={collider['shape']:<8s}  "
              f"constraint={constraint['type']}")
        annotated += 1

    return annotated, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Annotate glTF skeleton joints with tutorial physics extras.")
    parser.add_argument("input",  type=Path, help="Source .gltf file")
    parser.add_argument("output", type=Path, help="Destination .gltf file")
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional JSON config mapping bone names to extras "
                             "(overrides heuristics for named bones)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be annotated without writing output")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"ERROR: Input file not found: {args.input}")

    if args.input.suffix.lower() != ".gltf":
        sys.exit("ERROR: Only text-format .gltf files are supported "
                 "(not binary .glb). Extract with gltf-pipeline first.")

    with args.input.open(encoding="utf-8") as fh:
        gltf = json.load(fh)

    config: dict | None = None
    if args.config:
        with args.config.open(encoding="utf-8") as fh:
            config = json.load(fh)
        print(f"Using explicit config: {args.config}")

    print(f"\nAnnotating joints in: {args.input}")
    annotated, skipped = annotate(gltf, config, dry_run=args.dry_run)
    print(f"\n  {annotated} joints annotated, {skipped} skipped.")

    if args.dry_run:
        print("  Dry-run mode — no output file written.")
        return

    if annotated == 0:
        print("  Nothing to write.", file=sys.stderr)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(gltf, fh, indent=2, ensure_ascii=False)
    print(f"  Written: {args.output}")
    print()
    print("  Next steps:")
    print("    1. Run glTF-Validator to confirm the file is still valid.")
    print("    2. Open in your engine and verify collider shapes visually with DebugDrawer.")
    print("    3. Adjust masses/radii in the output file or create a --config JSON")
    print("       for per-bone overrides, then re-run this script.")


if __name__ == "__main__":
    main()
