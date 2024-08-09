#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
import gc
import pickle

import tbitk.ai.dl_cli as dl_cli
import os
import io
import tbparse

from pathlib import Path
from contextlib import redirect_stdout
from tbitk.ai.constants import AVAILABLE_LOSS_STRS


class KFoldModelMetadata:
    def __init__(self, hparams, all_models_save_root):
        self.hparams = hparams
        self.model_data_dir = self.get_model_data_dir_path(all_models_save_root)
        self.best_model_name = self.get_best_model_name()
        self.best_model_path = self.model_data_dir / self.best_model_name


    def get_model_data_dir_path(self, root):
        return Path(root) / self.hparams.name

    def get_best_model_name(self):
        return f"best_{self.hparams.name}.mha"

    def construct_training_cli_options(self):
        ret = [
            "--model_name", self.best_model_name,
            "--model_dir", str(self.model_data_dir),
            "--monitor_with_tb",
            "--logdir", str(self.model_data_dir),
        ]
        ret += self.hparams.construct_training_cli_options()
        return ret

class KFoldHParams:
    def __init__(self, encoder_name, pretrained, loss_str, **loss_fxn_kwargs):
        # Note that the keys in loss_fxn_kwargs must be cli options to dl_cli without
        # the leading '--'
        if loss_str not in AVAILABLE_LOSS_STRS:
            s = "loss_str must be one of " + " ".join(AVAILABLE_LOSS_STRS)
            raise ValueError(s)

        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.loss_str = loss_str
        self.loss_fxn_kwargs = loss_fxn_kwargs
        self.name = self.calculate_name()

    # Calling this __str__ seems wrong
    def calculate_name(self):
        ret = "pretrained" if self.pretrained else "non_pretrained"
        ret += f"_{self.encoder_name}_{self.loss_str}"
        for k, v in self.loss_fxn_kwargs.items():
            ret += f"_{k}_{v}"

        return ret

        # For example, pretrained_resnet34_focal_gamma_2

    def construct_training_cli_options(self):
        ret = [
            "--encoder_name", self.encoder_name,
            "--loss", self.loss_str,
        ]
        for key, val in self.loss_fxn_kwargs.items():
            ret.append(f"--{key}")
            ret.append(str(val))

        if not self.pretrained:
            ret.append("--no_pretrained")

        return ret


# This is the only cell that should need to change
#######
VIDEOS_DATA_DIR = Path("../../data/")
OUTPUT_DATA_DIR = Path("data/")
LOGFILE_PATH = OUTPUT_DATA_DIR / "log.txt"

file_patterns = [
    [
        str((VIDEOS_DATA_DIR / "training_head_phantom-20220121/annotation/**/A/*.pickle").resolve()),
        str((VIDEOS_DATA_DIR / "training_head_phantom-20220121/annotation/**/B/*.pickle").resolve()),
        str((VIDEOS_DATA_DIR / "training_head_phantom-20220121/annotation/**/E/*.pickle").resolve()),
    ],
    [
        str((VIDEOS_DATA_DIR / "unskilled_operator-20220421/annotation/**/[1-3]/*.pickle"))
    ],
    [
        str((VIDEOS_DATA_DIR / "unskilled_operator-20220421/annotation/**/[4-5]/*.pickle"))
    ],
    [
        str((VIDEOS_DATA_DIR / "unskilled_operator-20220421/annotation/**/[6-7]/*.pickle"))
    ],
    [
        str((VIDEOS_DATA_DIR / "unskilled_operator-20220421/annotation/**/[8-9]/*.pickle"))
    ]
]

K = 5
NUM_EPOCHS = 20
NUM_WORKERS = 4

hparams = [
    KFoldHParams("resnet34", False, "ce"),
    KFoldHParams("resnet34", True, "ce"),
]

#######

# Functions that call dl_cli
def run_extraction(train_patterns, val_patterns, test_patterns, output_dir, print_cmd=True):
    # Extract the data
    extract_cmd = [
        "extract",
        "--root_dir", str(output_dir.resolve()),
        "--train_patterns", *train_patterns,
        "--val_patterns", *val_patterns,
        "--test_patterns", *test_patterns,
        "--force"
    ]

    if print_cmd:
        print(extract_cmd)

    dl_cli.main(extract_cmd)

    return output_dir / "train", output_dir / "val", output_dir / "test"

def run_training(train_data_dirs, val_data_dirs, test_data_dirs, model_metadata, num_epochs=NUM_EPOCHS, num_workers=NUM_WORKERS,
                 print_cmd=True):
    train_cmd = [
        "train",
        "--train_data_dirs", *[str(p) for p in train_data_dirs],
        "--val_data_dirs", *[str(p) for p in val_data_dirs],
        "--test_data_dirs", *[str(p) for p in test_data_dirs],
        "--num_epochs", str(num_epochs),
        "--num_workers", str(num_workers),
    ]
    train_cmd += model_metadata.construct_training_cli_options()

    if print_cmd:
        print(train_cmd)

    dl_cli.main(train_cmd)

def timed_print(*args):
    print(f"{datetime.now()}:", *args)

assert len(file_patterns) == K
# The main models to be used. We'll make copies each fold,
# then train those copies
results = dict()
for fold_num in range(K):
    FOLD_OUTPUT_DATA_DIR = OUTPUT_DATA_DIR / str(fold_num)
    FOLD_OUTPUT_DATA_DIR.mkdir()
    FOLD_MODELS_DATA_DIR = FOLD_OUTPUT_DATA_DIR / "models"
    FOLD_MODELS_DATA_DIR.mkdir()

    metadatas = [KFoldModelMetadata(hparam, FOLD_MODELS_DATA_DIR) for hparam in hparams]
    results[fold_num] = dict()

    mode = "a" if fold_num else "w+"
    with open(str(LOGFILE_PATH), mode, buffering=1) as fp:
        with redirect_stdout(fp):
            timed_print("#" * 10, f"Starting {fold_num}", "#" * 10)
            # Extract
            test_index = fold_num
            val_index = (fold_num + 1) % len(file_patterns)

            test_patterns = file_patterns[test_index]
            val_patterns = file_patterns[val_index]
            train_patterns = []
            for i, p in enumerate(file_patterns):
                if i not in [test_index, val_index]:
                    train_patterns.extend(file_patterns[i])

            TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR = run_extraction(train_patterns, val_patterns, test_patterns, OUTPUT_DATA_DIR)

            # Rename keys
            os.rename(OUTPUT_DATA_DIR / "train_data_key.p", FOLD_OUTPUT_DATA_DIR / f"train_data_key_{fold_num}.p")
            os.rename(OUTPUT_DATA_DIR / "val_data_key.p", FOLD_OUTPUT_DATA_DIR / f"val_data_key_{fold_num}.p")
            os.rename(OUTPUT_DATA_DIR / "test_data_key.p", FOLD_OUTPUT_DATA_DIR / f"test_data_key_{fold_num}.p")

            # Run training, validation, and testing once for each model
            for model_metadata in metadatas:
                timed_print("Starting", model_metadata.hparams.name)
                run_training([TRAIN_DATA_DIR], [VAL_DATA_DIR], [TEST_DATA_DIR], model_metadata)

                gc.collect()
                gc.collect()

                df = tbparse.SummaryReader(str(model_metadata.model_data_dir)).scalars
                validation_df = df[df["tag"] == "Validation/dice_score"]
                idx_of_best_model = validation_df["value"].idxmax()
                step_of_best_model = validation_df.loc[idx_of_best_model, "step"]

                testing_df = df[df["tag"] == "Testing/dice_score"]
                best_model_test_dice = testing_df.loc[testing_df["step"] == step_of_best_model, "value"].iat[0]
                results[fold_num][model_metadata.hparams.name] = best_model_test_dice

                timed_print(model_metadata.hparams.name, "scored", best_model_test_dice, "testing dice")
                timed_print("Testing dice scores so far:", results)


with open(str(OUTPUT_DATA_DIR / "results.p"), "wb") as f:
    pickle.dump(results, f)

