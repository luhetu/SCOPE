import argparse
import os
import sys
import tempfile
import unittest

from utils.cfg import load_cfg


class LoadCfgTest(unittest.TestCase):
    def test_explicit_cli_overrides_yaml_values(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as cfg_file:
            cfg_file.write(
                "\n".join(
                    [
                        "task: seg",
                        "model: vit",
                        "data_dir: ./from_yaml",
                        "workers_per_gpu: 8",
                        "nowandb: false",
                    ]
                )
            )
            cfg_path = cfg_file.name

        old_argv = sys.argv
        try:
            sys.argv = [
                "prog",
                "--cfg",
                cfg_path,
                "--data_dir",
                "./from_cli",
                "--workers_per_gpu",
                "2",
                "--nowandb",
            ]
            parser = argparse.ArgumentParser()
            parser.add_argument("--cfg", type=str, default="")
            parser.add_argument("--data_dir", type=str, default=None)
            parser.add_argument("--workers_per_gpu", type=int, default=None)
            parser.add_argument("--nowandb", action="store_true")

            args = load_cfg(parser)
        finally:
            sys.argv = old_argv
            os.unlink(cfg_path)

        self.assertEqual(args.data_dir, "./from_cli")
        self.assertEqual(args.workers_per_gpu, 2)
        self.assertTrue(args.nowandb)
        self.assertEqual(args.model, "vit")
        self.assertEqual(args.task, "seg")


if __name__ == "__main__":
    unittest.main()
