import argparse
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.cfg import load_cfg


class LoadCfgOverridesTest(unittest.TestCase):
    def test_cli_overrides_are_preserved_after_yaml_load(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--cfg", default="")
        parser.add_argument("--model", default=None)
        parser.add_argument("--data_dir", default=None)
        parser.add_argument("--workers_per_gpu", type=int, default=None)

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as cfg_file:
            cfg_file.write(
                "\n".join(
                    [
                        "task: det",
                        "model: vit",
                        "data_dir: /yaml/coco",
                        "workers_per_gpu: 8",
                        "bs: 2",
                    ]
                )
            )
            cfg_path = cfg_file.name

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "prog",
                "--cfg",
                cfg_path,
                "--model",
                "vitscope",
                "--data_dir",
                "/cli/coco",
                "--workers_per_gpu",
                "2",
            ]
            args = load_cfg(parser)
        finally:
            sys.argv = old_argv
            os.remove(cfg_path)

        self.assertEqual(args.model, "vitscope")
        self.assertEqual(args.data_dir, "/cli/coco")
        self.assertEqual(args.workers_per_gpu, 2)
        self.assertEqual(args.task, "det")
        self.assertEqual(args.bs, 2)


if __name__ == "__main__":
    unittest.main()
