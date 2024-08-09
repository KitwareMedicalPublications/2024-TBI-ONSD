#!/usr/bin/env python

import os
import argparse

import tbitk.ai.deep_learning as dl
import torch
import monai

from tbitk.ai.constants import *
from tbitk.ai.inference_result import *
from tbitk.ai.transforms import (
    train_transforms,
    eval_transforms,
)
from tbitk.ai.deep_learning import get_focal_loss
from tbitk.util import videos_to_flat_list_of_frames
from torch import nn

def get_loss_from_str(s, *args, **kwargs):
    if s not in AVAILABLE_LOSS_STRS:
        raise ValueError("s must be \"ce\" or \"focal\"")
    if s == "ce":
        return nn.CrossEntropyLoss(*args, **kwargs)
    elif s == "focal":
        return get_focal_loss(*args, **kwargs)

def _construct_parser():
    """
    Constructs the ArgumentParser object with the appropriate options

    Returns
    ----------
    argparse.ArgumentParser
    """

    my_parser = argparse.ArgumentParser()
    sub_parsers = my_parser.add_subparsers(dest="sub_command")

    my_parser.add_argument("--show_alg_steps", action="store_true")

    # Next define some parent subparsers that will do our argument grouping.
    # This is to reduce code duplication for subcommands with the same args
    # https://stackoverflow.com/questions/33645859/how-to-add-common-arguments-to-argparse-subcommands

    # TODO: Below 2 args are the exact same except required.
    parent_parser_file_patterns_req = argparse.ArgumentParser(add_help=False)
    parent_parser_file_patterns_req.add_argument(
        "--file_patterns",
        action="store",
        nargs="+",
        help="Patterns to glob for",
        required=True,
    )

    parent_parser_file_patterns_no_req = argparse.ArgumentParser(add_help=False)
    parent_parser_file_patterns_no_req.add_argument(
        "--file_patterns",
        action="store",
        nargs="+",
        help="Patterns to glob for",
        required=False,
    )

    parent_parser_model_path = argparse.ArgumentParser(add_help=False)
    parent_parser_model_path.add_argument(
        "--model_path",
        action="store",
        help="Path to the model",
        required=True,
    )

    parent_parser_print_found_files = argparse.ArgumentParser(add_help=False)
    parent_parser_print_found_files.add_argument(
        "--print_found_files",
        action="store_true",
        help="Print the files found by globbing",
    )

    parent_parser_num_workers = argparse.ArgumentParser(add_help=False)
    parent_parser_num_workers.add_argument(
        "--num_workers",
        action="store",
        type=int,
        default=0,
        help="Num workers to use when loading frames.",
    )

    parent_parser_encoder_name = argparse.ArgumentParser(add_help=False)
    parent_parser_encoder_name.add_argument(
        "--encoder_name",
        action="store",
        type=str,
        default="resnet34",
        help="Encoder to use with the model. Defaults to resnet34",
    )


    # Now we start defining subcommands and their arguments
    sub_parser_extract = sub_parsers.add_parser(
        "extract",
        help="Extract video frames to train, val, and test subdirectories of root_dir."
        "By default, assumes subdirectories dont exists and creates them. See --force",
        parents=[parent_parser_file_patterns_no_req, parent_parser_print_found_files],
    )

    sub_parser_extract.add_argument(
        "--train_patterns",
        action="store",
        nargs="+",
        help="Glob patterns of the videos to use for training",
        required=False,
    )

    sub_parser_extract.add_argument(
        "--val_patterns",
        action="store",
        nargs="+",
        help="Glob patterns of the videos to use for validation",
        required=False,
    )

    sub_parser_extract.add_argument(
        "--test_patterns",
        action="store",
        nargs="+",
        help="Glob patterns of the videos to use for testing",
        required=False,
    )

    sub_parser_extract.add_argument(
        "--train_split",
        action="store",
        type=float,
        default=0.6,
        help="Percentage of total data to use for training."
        "1 - (train_split + val_split) is used for testing",
    )

    sub_parser_extract.add_argument(
        "--val_split",
        action="store",
        type=float,
        default=0.2,
        help="Percentage of total data to use for training."
        "1 - (train_split + val_split) is used for testing",
    )

    sub_parser_extract.add_argument(
        "--no_print_split",
        action="store_true",
        help="Dont print the split for the train, val, and test files",
    )

    sub_parser_extract.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the train, val, and test subdirs if already present",
    )

    sub_parser_extract.add_argument(
        "--root_dir",
        action="store",
        help="Root directory to extract to",
        type=path.Path,
        required=True,
    )

    sub_parser_extract.add_argument(
        "--dont_map_images_to_source",
        action="store_true",
        help="Dont write out a dictionary mapping extracted frames to"
        "original source file name",
    )

    sub_parser_train = sub_parsers.add_parser(
        "train",
        help="Train a fresh model",
        parents=[parent_parser_num_workers, parent_parser_encoder_name],
    )


    sub_parser_train.add_argument(
         "--loss",
        choices=AVAILABLE_LOSS_STRS,
        default="ce",
        help="Loss function to use. Use \"ce\" for cross entropy loss, and \"focal\" for focal loss",
    )

    sub_parser_train.add_argument(
        "--gamma",
        type=float,
        help="Gamma value if using focal loss. Has no effect otherwise."
    )

    sub_parser_train.add_argument(
         "--no_pretrained",
        action="store_true",
        help="Don't use a model pretrained on imagenet",
    )

    sub_parser_train.add_argument(
         "--train_data_dirs",
        action="store",
        nargs="+",
        help="Path to the training data directories",
        required=True,
    )

    sub_parser_train.add_argument(
         "--val_data_dirs",
        action="store",
        nargs="+",
        help="Path to the validation data directories",
        required=True,
    )

    sub_parser_train.add_argument(
         "--test_data_dirs",
        action="store",
        nargs="+",
        help="Path to the test data directories. Can monitor the test accuracy during training",
        required=False,
    )

    sub_parser_train.add_argument(
        "--calculate_weights",
        action="store_true",
        help="Calculate weights for the current training data",
    )

    sub_parser_train.add_argument(
        "--model_name",
        action="store",
        help="Name of the best model",
        default=str(DEFAULT_BEST_MODEL_NAME)
    )

    sub_parser_train.add_argument(
        "--model_dir",
        action="store",
        type=path.Path,
        help="Directory to save the best model to",
        required=True,
    )

    sub_parser_train.add_argument(
        "--monitor_with_tb",
        action="store_true",
        help="Monitor model training with tensorboard.",
    )

    sub_parser_train.add_argument(
        "--logdir",
        action="store",
        help="logdir for tensorboard",
    )

    sub_parser_train.add_argument(
        "--num_epochs",
        action="store",
        help="Number of epochs to train for.",
        default=DEFAULT_NUM_EPOCHS,
        type=int,
    )


    sub_parser_test = sub_parsers.add_parser(
        "test",
        help="Test a trained model",
        parents=[parent_parser_model_path, parent_parser_num_workers, parent_parser_encoder_name]
    )

    sub_parser_test.add_argument(
         "--test_data_dirs",
        action="store",
        nargs="+",
        help="Path to the testing data directories",
        required=True,
    )

    sub_parser_display_mask = sub_parsers.add_parser(
        "display_mask",
        help="Display a mask at the given path.",
    )

    sub_parser_display_mask.add_argument(
        "mask_path",
        action="store",
        type=str,
        help="Path to a mask to display.",
    )

    sub_parser_predict_image = sub_parsers.add_parser(
        "predict_image",
        help="Run inference on an image",
        parents=[parent_parser_model_path, parent_parser_encoder_name]
    )

    sub_parser_predict_image.add_argument(
        "--image_path",
        action="store",
        type=str,
        help="Path to an input image.",
        required=True,
    )

    sub_parser_predict_video = sub_parsers.add_parser(
        "predict_video",
        help="Predict the ONSD a video. If multiple videos are specified,"
        "combines the sources into one video.",
        parents=[parent_parser_file_patterns_req, parent_parser_model_path, parent_parser_encoder_name],

    )
    sub_parser_predict_video.add_argument(
        "--animate",
        action="store_true",
        help="Collect predictions for each frame of specified videos into an"
        "animation. By default, shows animation on screen. If"
        "--animation_save_path is specified, saves to disk instead.",
    )

    sub_parser_predict_video.add_argument(
        "--fps",
        action="store",
        type=float,
        help="FPS of the resulting animation.",
        default=24
    )

    sub_parser_predict_video.add_argument(
        "--actual_onsd",
        action="store",
        type=int,
        help="The actual ONSD for a prediction",
    )

    sub_parser_predict_video.add_argument(
        "--animation_save_path",
        action="store",
        help="Path to save the animation of the models predictions to.",
    )

    sub_parser_save_measurements_to_json = sub_parsers.add_parser(
        "save_measurements_to_json",
        help="Loads videos matching the specified patterns and saves the onsd estimation as json",
        parents=[parent_parser_file_patterns_req],
    )


    sub_parser_save_measurements_to_json.add_argument(
        "--keep_file_structure",
        action="store_true",
        help="When saving the measurements, use the same relative directory structure of the input files.",

    )

    sub_parser_save_inference_results = sub_parsers.add_parser(
        "save_inference_results",
        help="Loads videos matching the specified patterns and saves the inference results to disk",
        parents=[parent_parser_file_patterns_req, parent_parser_model_path, parent_parser_print_found_files, parent_parser_encoder_name],
    )

    sub_parser_save_inference_results.add_argument(
        "--root_dir",
        action="store",
        type=path.Path,
        help="Root directory to save the InferenceResult objects to",
        required=True
    )

    group = sub_parser_save_inference_results.add_mutually_exclusive_group()
    group.add_argument(
        "--force",
        action="store_true",
        help="Clear the root_dir if present",
    )

    group.add_argument(
        "--keep_root",
        action="store_true",
        help="Keep the root directory, if already present, only overwriting overlapping files",
    )

    sub_parser_save_inference_results.add_argument(
        "--keep_file_structure",
        action="store_true",
        help="When saving the inference results, use the same relative directory structure of the input files.",
    )

    sub_parser_save_inference_results.add_argument(
        "--combine_sources",
        action="store_true",
        help="Combine the source images / video into one source and save inference results on that source.",
    )

    sub_parser_save_inference_results.add_argument(
        "--combined_res_name",
        action="store",
        type=str,
        default="combined_inf_res",
        help="Name of the combined inference result"
    )

    sub_parser_load_and_display_inference_result = sub_parsers.add_parser(
        "load_and_display_inference_result",
        help="Load the inference result at the path and display the source and prediction, either as an image if 2d or an animation if 3d",
    )

    sub_parser_load_and_display_inference_result.add_argument(
        "result_path",
        action="store",
        help="path to inference result to load"
    )

    sub_parser_load_and_display_inference_result.add_argument(
        "--fps",
        action="store",
        type=float,
        help="FPS of the resulting animation.",
        default=24
    )
    return my_parser


def main(args=None):
    monai.utils.misc.set_determinism(46842)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_parser = _construct_parser()
    args = my_parser.parse_args(args=args)

    if args.sub_command == "extract":
        individual_patterns = [args.train_patterns, args.val_patterns, args.test_patterns]
        if individual_patterns.count(None) not in [0, 3]:
            msg = "Either all 3 of train_patterns, val_patterns, and test_patterns must be specified or not"
            my_parser.error(msg)

        if args.train_patterns and args.file_patterns:
            msg = "train_patterns, val_patterns, and test_patterns are all mutually exclusive with file_patterns"
            my_parser.error(msg)

        if not (args.train_patterns or args.file_patterns):
            msg = "Either specify all of train, val, and test patterns, or file_patterns"
            my_parser.error(msg)

        train_img_fnames, train_mask_fnames = None, None
        val_img_fnames, val_mask_fnames = None, None
        test_img_fnames, test_mask_fnames = None, None
        if args.file_patterns:
            xpaths, ypaths = dl.get_fnames(args.file_patterns)
            assert len(xpaths) != 0, "No filepaths found"
            if args.print_found_files:
                for xp, yp in zip(xpaths, ypaths):
                    print(f"Source file: {xp}\nannotation file {yp}", end="\n\n")

            (
                train_img_fnames,
                train_mask_fnames,
                val_img_fnames,
                val_mask_fnames,
                test_img_fnames,
                test_mask_fnames,
            ) = dl.split_filenames(xpaths, ypaths, args.train_split, args.val_split, shuffle=True)
        elif args.train_patterns:
            train_img_fnames, train_mask_fnames = dl.get_fnames(args.train_patterns)
            val_img_fnames, val_mask_fnames = dl.get_fnames(args.val_patterns)
            test_img_fnames, test_mask_fnames = dl.get_fnames(args.test_patterns)

        if not args.no_print_split:
            print("Train filenames:")
            for train_img, train_mask in zip(train_img_fnames, train_mask_fnames):
                print(f"\ttrain image: {train_img}\n\tannotation file {train_mask}", end="\n\n")
            print("val filenames:")
            for val_img, val_mask in zip(val_img_fnames, val_mask_fnames):
                print(f"\tval image: {val_img}\n\tannotation file {val_mask}", end="\n\n")
            print("test filenames:")
            for test_img, test_mask in zip(test_img_fnames, test_mask_fnames):
                print(f"\ttest image: {test_img}\n\tannotation file {test_mask}", end="\n\n")

        dl.extract_data(
            train_img_fnames,
            train_mask_fnames,
            val_img_fnames,
            val_mask_fnames,
            test_img_fnames,
            test_mask_fnames,
            args.root_dir,
            overwrite_dirs=args.force,
            map_images_to_source=not args.dont_map_images_to_source,
        )

    if args.sub_command == "display_mask":
        mask = itk.imread(args.mask_path)

        dl.display_image_truth_and_prediction(truth=mask)

    if args.sub_command == "train":
        weights = None

        if args.calculate_weights:
            if len(args.train_data_dirs) > 1:
                msg = "Weight calculation not implemented for multiple train dirs"
                raise RuntimeError(msg)
            weights = dl.calculate_weights(args.train_data_dirs[0])
            print(weights)

        model = dl.get_model(use_pretrained_weights=not args.no_pretrained, encoder_name=args.encoder_name)

        train_loader = dl.get_data_loader(
            args.train_data_dirs, train_transforms, num_workers=args.num_workers
        )
        val_loader = dl.get_data_loader(
            args.val_data_dirs, eval_transforms, num_workers=args.num_workers
        )

        test_loader = None
        if args.test_data_dirs:
            test_loader = dl.get_data_loader(
                args.test_data_dirs, eval_transforms, num_workers=args.num_workers
            )

        if weights is not None:
            weights = torch.Tensor(weights).to(device)

        loss_kwargs = dict()
        if args.loss == "ce":
            loss_kwargs["weight"] = weights
        elif args.loss == "focal":
            loss_kwargs["alpha"] = weights
            loss_kwargs["gamma"] = args.gamma

        loss_function = get_loss_from_str(args.loss, **loss_kwargs)

        val_losses = dl.train_model(
            model,
            train_loader,
            val_loader,
            device,
            args.model_name,
            args.model_dir,
            loss_function=loss_function,
            monitor_with_tb=args.monitor_with_tb,
            tb_logdir=args.logdir,
            num_epochs=args.num_epochs,
            test_loader=test_loader,
        )
        print(val_losses)

    if args.sub_command == "test":
        model = dl.load_model(args.model_path, encoder_name=args.encoder_name)
        test_loader = dl.get_data_loader(
            args.test_data_dirs, eval_transforms, num_workers=args.num_workers
        )
        dl.test_model(model, test_loader, device)

    if args.sub_command == "predict_image":
        model = dl.load_model(args.model_path, encoder_name=args.encoder_name)
        dl.predict_image_and_show(model,  args.image_path, device, show_alg_steps=args.show_alg_steps)

    if args.sub_command == "predict_video":
        model = dl.load_model(args.model_path, encoder_name=args.encoder_name)
        filepaths = dl.get_filepaths_matching_pattern(args.file_patterns, recursive=True)

        input_videos_and_images = [itk.imread(fp).astype(IMAGE_PIXEL_TYPE) for fp in filepaths]
        if len(input_videos_and_images) == 1:
            input_ = input_videos_and_images[0]
        else:
            input_ = videos_to_flat_list_of_frames(input_videos_and_images)

        res_3d = dl.run_inference(model, input_, device=device, show_alg_steps=args.show_alg_steps)
        print("Final onsd for video:", res_3d.onsd)
        if args.animate:
            dl.generate_prediction_animation(
                res_3d, args.animation_save_path, actual_onsd=args.actual_onsd, fps=args.fps
            )

    if args.sub_command == "save_inference_results":
        save_path_root = args.root_dir
        if save_path_root.exists() and not args.keep_root:
            if args.force:
                shutil.rmtree(str(save_path_root))
            else:
                err_msg = f"{str(save_path_root)} already exists. Please" \
                " either remove the directory or specify one of --force or --keep_root."
                raise FileExistsError(err_msg)

        assert((not save_path_root.exists()) or (save_path_root.exists() and args.keep_root))
        if not args.keep_root:
            save_path_root.mkdir(parents=True)

        filepaths = sorted(dl.get_filepaths_matching_pattern(args.file_patterns, recursive=True))
        if args.print_found_files:
            print("\n".join(filepaths))
        model = dl.load_model(args.model_path, encoder_name=args.encoder_name)
        # TODO: mutually exclusive args with combine_sources and keep_file_structure
        if args.combine_sources:
            input_videos_and_images = [itk.imread(fp).astype(IMAGE_PIXEL_TYPE) for fp in filepaths]
            input_ = videos_to_flat_list_of_frames(input_videos_and_images)
            res = dl.run_inference(model, input_, device=device)
            save_path = save_path_root / args.combined_res_name
            res.save_to_dir(save_path)
            print(f"\tSaved to {save_path}", end="\n\n")
        else:
            commonpath = None
            if args.keep_file_structure and len(filepaths) > 1:
                commonpath = os.path.commonpath(filepaths)

            for fp in filepaths:
                input_ = itk.imread(fp, IMAGE_PIXEL_TYPE)
                res = dl.run_inference(model, input_, device=device)

                save_path = save_path_root
                if commonpath is not None:
                    print(f"Predicted onsd for {path.Path(fp).relative_to(commonpath)}: {res.onsd}")
                    parent_dir = path.Path(fp).parents[0]
                    save_path = save_path_root / parent_dir.relative_to(commonpath)
                else:
                    print(f"Predicted onsd for {path.Path(fp).name}: {res.onsd}")

                save_path /= (path.Path(fp).stem + "_inf_res")
                res.save_to_dir(save_path)
                print(f"\tSaved to {save_path}", end="\n\n")

    # For now, this only works for one inference result at a time.
    if args.sub_command == "load_and_display_inference_result":
        res = InferenceResult.load_from_dir(args.result_path)
        if isinstance(res, InferenceResult2D):
            dl.display_image_truth_and_prediction(image=res.source, prediction=res.mask)

        if isinstance(res, InferenceResult3D):
            dl.generate_prediction_animation(res, None, fps=args.fps)




if __name__ == "__main__":
    main()
