import tempfile
import unittest
import tbitk.ai.dl_cli as dl_cli
import tbitk.ai.deep_learning as dl
import gc

from pathlib import Path


class TestTrain(unittest.TestCase):
    WORKING_DIR = Path(__file__).parents[0]
    AI_TEST_DIR = Path(__file__).parents[1]

    A1_VIDEO_PATH = (AI_TEST_DIR / "data/preprocessed/a-1.mha").resolve()
    B2_VIDEO_PATH = (AI_TEST_DIR / "data/preprocessed/b-2.mha").resolve()
    E1_VIDEO_PATH = (AI_TEST_DIR / "data/preprocessed/e-1.mha").resolve()

    MODEL_NAME = "test_model.pt"
    NUM_EPOCHS = 1
    # TODO: best place to put this?
    unet_name_format = "u-{}"

    @classmethod
    def setUpClass(cls):
        cls.tdir = tempfile.TemporaryDirectory()
        cls.EXTRACT_PATH = Path(cls.tdir.name) / "extracted_data"
        cls.TRAIN_DATA_DIR = cls.EXTRACT_PATH / "train"
        cls.VAL_DATA_DIR = cls.EXTRACT_PATH / "val"
        cls.TEST_DATA_DIR = cls.EXTRACT_PATH / "test"
        cls.MODEL_DIR = Path(cls.tdir.name) / "models"
        cls.LOGDIR = Path(cls.tdir.name) / "runs"

        # Now construct cli args
        args = [
            "extract",
            "--root_dir",
            str(cls.EXTRACT_PATH),
            "--train_patterns",
            str(cls.A1_VIDEO_PATH),
            "--val_patterns",
            str(cls.B2_VIDEO_PATH),
            "--test_patterns",
            str(cls.E1_VIDEO_PATH),
            "--print_found_files",
        ]

        dl_cli.main(args)

    @classmethod
    def tearDownClass(cls):
        cls.tdir.cleanup()


    # Pytorch has some memory leaks which effect the gpu
    # Just clear out the garbage collector to be safe
    def tearDown(self) -> None:
        gc.collect()
        gc.collect()

    def call_train(
        self,
        train_data_dir=None,
        val_data_dir=None,
        model_name=None,
        model_dir=None,
        logdir=None,
        num_epochs=None,
        encoder_name=None,
    ):
        train_data_dir = train_data_dir or self.TRAIN_DATA_DIR
        val_data_dir = val_data_dir or self.VAL_DATA_DIR
        model_name = model_name or self.MODEL_NAME
        model_dir = model_dir or self.MODEL_DIR
        logdir = logdir or self.LOGDIR
        num_epochs = num_epochs or self.NUM_EPOCHS
        args = [
            "train",
            "--train_data_dirs",
            str(train_data_dir),
            "--val_data_dirs",
            str(val_data_dir),
            "--model_name",
            str(model_name),
            "--model_dir",
            str(model_dir),
            "--logdir",
            str(logdir),
            "--num_epochs",
            str(num_epochs),
            "--monitor_with_tb",
        ]

        if encoder_name:
            args.append("--encoder_name")
            args.append(str(encoder_name))

        dl_cli.main(args)

    def test_train_specify_encoder(self):
        encoder_names = ["resnet34", "timm-regnety_002"]
        for encoder_name in encoder_names:
            self.call_train(encoder_name=encoder_name)
            # Technically this below call will error if the encoder name doesn't match,
            # watch makes the assert below it redundant, but it's still good to have.
            model = dl.load_model(
                self.MODEL_DIR / self.MODEL_NAME, encoder_name=encoder_name
            )
            self.assertEqual(model.name, self.unet_name_format.format(encoder_name))

    def test_train_default_encoder(self):
        self.call_train()
        model = dl.load_model(self.MODEL_DIR / self.MODEL_NAME)
        self.assertEqual(model.name, self.unet_name_format.format("resnet34"))
