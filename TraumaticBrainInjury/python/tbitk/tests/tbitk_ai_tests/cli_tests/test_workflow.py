import io
import tempfile
import unittest
import tbitk.ai.dl_cli as dl_cli

from pathlib import Path
from contextlib import redirect_stdout


class TestWorkflow(unittest.TestCase):
    WORKING_DIR = Path(__file__).parents[0]
    AI_TEST_DIR = Path(__file__).parents[1]

    A1_VIDEO_PATH = (AI_TEST_DIR / "data/preprocessed/a-1.mha").resolve()
    A2_VIDEO_PATH = (AI_TEST_DIR / "data/preprocessed/a-2.mha").resolve()
    B2_VIDEO_PATH = (AI_TEST_DIR / "data/preprocessed/b-2.mha").resolve()
    E1_VIDEO_PATH = (AI_TEST_DIR / "data/preprocessed/e-1.mha").resolve()

    # Note that the below paths are meant to be relative to
    # the tempdir we create
    EXTRACT_SUBDIR = Path("extract_root")
    TRAIN_SUBDIR = EXTRACT_SUBDIR / "train"
    VAL_SUBDIR = EXTRACT_SUBDIR / "val"
    TEST_SUBDIR = EXTRACT_SUBDIR / "test"

    MODEL_NAME = "test_model.pt"
    MODEL_SUBDIR = Path("models")
    MODEL_PATH = MODEL_SUBDIR / MODEL_NAME

    LOGSUBDIR = "runs"

    INF_RES_SUBDIR = Path("inference_results")
    E1_INF_RES_PATH = INF_RES_SUBDIR / "e-1_inf_res"


    def _extract_and_verify(self, tdir):
        EXTRACT_PATH = tdir / self.EXTRACT_SUBDIR
        # Extract
        self.assertFalse(EXTRACT_PATH.exists())

        # Now construct cli args
        args = [
            "extract",
            "--root_dir", str(EXTRACT_PATH),
            "--train_patterns", str(self.A1_VIDEO_PATH), str(self.A2_VIDEO_PATH),
            "--val_patterns", str(self.B2_VIDEO_PATH),
            "--test_patterns", str(self.E1_VIDEO_PATH),
            "--print_found_files",
        ]

        dl_cli.main(args)

        # In test_extract, we check that the number of frames written out is correct.
        # Here we'll just check that the directories exist
        for subdir in [self.TRAIN_SUBDIR, self.VAL_SUBDIR, self.TEST_SUBDIR]:
            PATH_TO_SUBDIR = tdir / subdir
            self.assertTrue(PATH_TO_SUBDIR.exists())

            img_patterns = list(PATH_TO_SUBDIR.glob("img*"))
            mask_patterns = list(PATH_TO_SUBDIR.glob("mask*"))

            self.assertEqual(len(mask_patterns), len(img_patterns))
            self.assertNotEqual(len(img_patterns), 0)

    def _train_model_and_verify(self, tdir):
        self.assertFalse((tdir / self.MODEL_PATH).exists())
        self.assertFalse((tdir / self.LOGSUBDIR).exists())
        args = [
            "train",
            "--train_data_dirs", str(tdir / self.TRAIN_SUBDIR),
            "--val_data_dirs", str(tdir / self.VAL_SUBDIR),
            "--model_name", str(self.MODEL_NAME),
            "--model_dir", str(tdir / self.MODEL_SUBDIR),
            "--monitor_with_tb",
            "--logdir", str(tdir / self.LOGSUBDIR),
            "--num_epochs", "10",
        ]

        dl_cli.main(args)

        self.assertTrue((tdir / self.MODEL_PATH).exists())
        self.assertTrue((tdir / self.LOGSUBDIR).exists())


    def _test_model_and_verify(self, tdir):
        args = [
            "test",
            "--test_data_dirs", str(tdir / self.TEST_SUBDIR),
            "--model_path", str(tdir / self.MODEL_PATH),
        ]
        f = io.StringIO()
        with redirect_stdout(f):
            dl_cli.main(args)

        captured_test_acc = float(f.getvalue())
        self.assertGreaterEqual(captured_test_acc, 0.7)

    def _save_inf_res_and_verify(self, tdir):
        self.assertFalse((tdir / self.E1_INF_RES_PATH).exists())

        args = [
            "save_inference_results",
            "--keep_file_structure",
            "--file_patterns", str(self.E1_VIDEO_PATH),
            "--print_found_files",
            "--model_path", str(tdir / self.MODEL_PATH),
            "--root_dir", str(tdir / self.INF_RES_SUBDIR),
        ]

        dl_cli.main(args)

        self.assertTrue((tdir / self.E1_INF_RES_PATH).exists())

    def test_workflow(self):
        with tempfile.TemporaryDirectory() as tdir:
            tdir_path = Path(tdir)
            # TODO: Old example had visualization code... some way to test that?
            self._extract_and_verify(tdir_path)
            self._train_model_and_verify(tdir_path)
            self._test_model_and_verify(tdir_path)
            self._save_inf_res_and_verify(tdir_path)


if __name__ == "__main__":
    unittest.main()
