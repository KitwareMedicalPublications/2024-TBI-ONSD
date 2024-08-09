import tempfile
import unittest
import tbitk.ai.dl_cli as dl_cli

from pathlib import Path


class TestSaveInferenceResult(unittest.TestCase):
    AI_TEST_DIR = Path(__file__).parents[1]
    A1_SMALL_VIDEO_PATH = (AI_TEST_DIR / "data/preprocessed/a-1_small.mha").resolve()
    A2_SMALL_VIDEO_PATH = (AI_TEST_DIR / "data/preprocessed/a-2_small.mha").resolve()

    MODEL_PATH = str((AI_TEST_DIR / "models/example_model.pt").resolve())
    INF_RES_SUBDIR = Path("inference_results")

    def _save_inf_res_cli(self, inf_res_root, *args, file_patterns=None):
        if file_patterns is None:
            file_patterns = str(self.A1_SMALL_VIDEO_PATH)
        cli_args = [
                "save_inference_results",
                "--file_patterns", str(file_patterns),
                "--print_found_files",
                "--model_path", str(self.MODEL_PATH),
                "--root_dir", str(inf_res_root),
                *args
            ]
        dl_cli.main(cli_args)


    def test_save_inf_res(self):
        with tempfile.TemporaryDirectory() as tdir:
            inf_res_root = Path(tdir) / self.INF_RES_SUBDIR
            self.assertFalse(inf_res_root.exists())

            self._save_inf_res_cli(inf_res_root)

            self.assertEqual(len(list(inf_res_root.glob("a-1_small_inf_res/"))), 1)

    def test_save_inf_res_already_exist(self):
        with tempfile.TemporaryDirectory() as tdir:
            inf_res_root = Path(tdir) / self.INF_RES_SUBDIR
            inf_res_root.mkdir(exist_ok=False)

            self.assertRaises(FileExistsError, self._save_inf_res_cli, inf_res_root)

            self.assertTrue(inf_res_root.exists())

    def test_save_inf_res_force(self):
        with tempfile.TemporaryDirectory() as tdir:
            inf_res_root = Path(tdir) / self.INF_RES_SUBDIR

            # Generate inference result for a2
            self._save_inf_res_cli(inf_res_root, file_patterns=self.A2_SMALL_VIDEO_PATH)

            self.assertTrue(inf_res_root.exists())

            # Now go for a-1 but specify force
            self._save_inf_res_cli(inf_res_root, "--force")

            # a-2 should be gone, but a-1 should be there
            self.assertEqual(len(list(inf_res_root.glob("a-2_small_inf_res/"))), 0)
            self.assertEqual(len(list(inf_res_root.glob("a-1_small_inf_res/"))), 1)

    def test_save_inf_res_keep_root_dir(self):
        with tempfile.TemporaryDirectory() as tdir:
            inf_res_root = Path(tdir) / self.INF_RES_SUBDIR

            # Generate inference result for a2
            self._save_inf_res_cli(inf_res_root, file_patterns=self.A2_SMALL_VIDEO_PATH)

            self.assertTrue(inf_res_root.exists())

            # Now go for a-1 but specify keep_root_dir
            self._save_inf_res_cli(inf_res_root, "--keep_root")

            # a-1 and a-2 should be there
            self.assertEqual(len(list(inf_res_root.glob("a-2_small_inf_res/"))), 1)
            self.assertEqual(len(list(inf_res_root.glob("a-1_small_inf_res/"))), 1)
