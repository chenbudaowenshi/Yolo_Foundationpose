"""农夫山泉、东方树叶接入：类别和 mesh 必须对应同一个瓶子。"""
import contextlib
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

os.environ["D435I_FOPOSE_REEXEC"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d435i_foundationpose as base
from d435i_autonomous_vision import build_parser as autonomous_parser
from sb_pose_contract import load_object_profile, profile_target_class

ROOT = Path(__file__).resolve().parents[2]
NAMES = {0: "fenda", 1: "cola", 2: "qixi", 3: "nfsq", 4: "dfsy"}


class ObjectArgumentTests(unittest.TestCase):
    def resolve(self, *argv, parser=None):
        parser = parser or base.build_parser()
        args = parser.parse_args(list(argv))
        base.resolve_object_arguments(args, parser)
        return args

    def assert_rejected(self, *argv):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.resolve(*argv)

    def test_defaults_keep_old_bottle(self):
        args = self.resolve()
        self.assertEqual((args.target_class, args.mesh, args.object), (0, base.DEFAULT_MESH, None))

    def test_object_selects_class_and_mesh(self):
        for name, class_id in (("nfsq", 3), ("dfsy", 4)):
            args = self.resolve("--object", name)
            self.assertEqual(args.target_class, class_id)
            self.assertEqual(args.mesh, ROOT / f"runs/mesh/{name}/{name}.obj")

    def test_class_alone_selects_matching_mesh(self):
        args = self.resolve("--target-class", "4")
        self.assertEqual((args.object, args.mesh), ("dfsy", ROOT / "runs/mesh/dfsy/dfsy.obj"))
        self.assertEqual(self.resolve("--target-class", "1").mesh, base.DEFAULT_MESH)

    def test_explicit_mesh_is_kept(self):
        args = self.resolve("--target-class", "3", "--mesh", "/tmp/custom.obj")
        self.assertEqual((args.object, args.mesh), (None, Path("/tmp/custom.obj")))

    def test_conflicts_are_rejected(self):
        self.assert_rejected("--object", "nfsq", "--target-class", "4")
        self.assert_rejected("--object", "nfsq", "--mesh", str(ROOT / "runs/mesh/bottle.obj"))

    def test_autonomous_parser_has_object(self):
        args = self.resolve("--object", "nfsq", parser=autonomous_parser())
        self.assertEqual(args.target_class, 3)

    def test_detector_class_names_must_match(self):
        args = NS(target_class=3, object="nfsq")
        base.check_target_class(NS(names=NAMES), args)
        with self.assertRaises(ValueError):
            base.check_target_class(NS(names={0: "fenda", 1: "cola", 2: "qixi", 3: "other"}), args)
        with self.assertRaises(ValueError):
            base.check_target_class(NS(names={0: "fenda", 1: "cola", 2: "qixi"}), args)


class ProfileTests(unittest.TestCase):
    def test_profiles_match_meshes(self):
        for name, class_id in (("nfsq", 3), ("dfsy", 4)):
            profile = load_object_profile(ROOT / f"my_demo/config/{name}.json")
            self.assertEqual((profile["object_id"], profile["class_id"]), (name, class_id))
            self.assertEqual(Path(profile["mesh_path"]), ROOT / f"runs/mesh/{name}/{name}.obj")
            height = profile["expected_extents_m"][2]
            self.assertAlmostEqual(profile["T_model_object_center"][2][3], height / 2, delta=1e-6)

    def test_profile_target_class(self):
        bottle = load_object_profile(ROOT / "my_demo/config/bottle_current.json")
        nfsq = load_object_profile(ROOT / "my_demo/config/nfsq.json")
        self.assertEqual(profile_target_class(bottle), 0)
        self.assertEqual(profile_target_class(bottle, 2), 2)
        self.assertEqual(profile_target_class(nfsq), 3)
        self.assertEqual(profile_target_class(nfsq, 3), 3)
        with self.assertRaises(ValueError):
            profile_target_class(nfsq, 0)

    def test_ros_node_default_class_follows_profile(self):
        from sb_vision_node import build_parser as node_parser
        args = node_parser().parse_args([])
        profile = load_object_profile(args.object_profile)
        self.assertEqual((profile["object_id"], profile_target_class(profile, args.target_class)), ("dfsy", 4))


if __name__ == "__main__":
    unittest.main()
