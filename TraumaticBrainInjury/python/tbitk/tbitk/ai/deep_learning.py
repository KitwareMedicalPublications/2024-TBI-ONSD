# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Some sections below are taken from the monai examples found at
# https://github.com/Project-MONAI/tutorials/blob/master/2d_segmentation/torch/unet_training_array.py
# https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/brats_segmentation_3d.ipynb
# https://github.com/Project-MONAI/MONAI/blob/af88eebe813e8576eca8f036731cf48012909457/monai/transforms/utility/array.py
# https://github.com/Project-MONAI/MONAIBootcamp2021/blob/main/day1/3.%20End-To-End%20Workflow%20with%20MONAI.ipynb


# TODO: restructure this file a bit. Create a separate dataset, file for the transform, etc.

# Master todo
# TODO: reformat with black or similar
# TODO: remove get_slice from util
# TODO: type hints

import pathlib as path

import torch
import pathlib as path
import numpy as np
import monai
import itk
import matplotlib.pyplot as plt
import sklearn.utils
import shutil
import segmentation_models_pytorch as smp
import matplotlib.animation as animation

import tbitk.data_manager as dm

from glob import glob
from ignite.engine import Events
from ignite.metrics import Loss
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from monai.data import Dataset, DataLoader
from monai.transforms import (
    AddChannel,
    AsDiscrete,
    Compose,
    Resize,
)
from tbitk.util import (
    extract_slice,
    itk_image_is_2d,
    itk_image_is_video,
)
from matplotlib import cm
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import ValidationHandler, MeanDice, CheckpointSaver
from monai.metrics import compute_meandice
from natsort import natsorted
from tbitk.ocularus import aggregate_onsds, calculate_onsd_from_mask, image_from_array, ONSDDebugInfo
# TODO: Make deep_learning.py and related things a package
from tbitk.ai.transforms import *
from tbitk.ai.constants import *
from tbitk.ai.inference_result import *



def get_model(use_pretrained_weights=True, encoder_name="resnet34"):
    """
    Loads a fresh model pretrained with imagenet params.

    Parameters
    ----------
    encoder_name : str
        Name of the encoder to use. See segmentation_models_pytorch docs

    Returns
    ----------
    torch.nn.Module
    """
    return smp.Unet(
        encoder_name=encoder_name, encoder_weights="imagenet" if use_pretrained_weights else None, in_channels=1,
        classes=3, activation=None
    )

def get_focal_loss(*args, gamma=None, reduction="mean", **kwargs):
    if gamma is None:
        gamma = 2
    return torch.hub.load(
        'adeelh/pytorch-multi-class-focal-loss',
        *args,
        model='FocalLoss',
        reduction=reduction,
        gamma=gamma,
        **kwargs,
    )




# TODO: Could use a different name
def get_fnames(patterns):
    """
    Given file patterns (globs) matching preprocessed images, returns paths
    to the images and the corresponding annotations.
    This is done using tbitk.data_manager.

    Parameters
    ----------
    patterns : list
        The glob patterns to match when searching for files
    rel_to_input_dir : bool
        Treat relative paths as relative to INPUT_DATA_DIR
        defined in constants.py

    Returns
    -------
    image_paths : list of str
    eye_and_nerve_annotation_paths : list of str
    """
    annotation_paths = []
    for pattern in patterns:
        annotation_paths.extend(glob(pattern, recursive=True))
    annotation_paths = sorted(annotation_paths)
    image_paths = []
    eye_and_nerve_annotation_paths = []
    for ap in annotation_paths:
        related_filepaths = dm.get_filepaths(ap)
        preprocessed_path = path.Path(related_filepaths["preprocessed"])
        annotation_path = path.Path(related_filepaths["annotation_label_image"])

        image_paths.append(str(preprocessed_path.resolve()))
        eye_and_nerve_annotation_paths.append(str(annotation_path.resolve()))

    return image_paths, eye_and_nerve_annotation_paths


def image_and_mask3d_to_list_of_unique_frames(img, mask):
    """
    Turns an img and mask into two lists of unique frames.
    Which frames are unique are determined by the source image. i.e,
    two unique source frames may have the same mask. In which case,
    Both source frames and mask frames would be considered unique.
    If the image and mask are 2D, just two lists, the first containing
    the image, the second containing the mask

    Parameters
    ----------
    img : itk.Image[itk.F, 2] or itk.Image[itk.F, 3]
        2D or 3D image representing a video
    mask : itk.Image[itk.UC, 2] or itk.Image[itk.UC, 3]
        2D or 3D image representing the masks for each frame of img

    Returns
    -------
    image_frames : list of itk.Image[itk.F, 2]
    mask_frames : list of itk.Image[itk.UC, 2]
    """
    # Assumes img and mask have the same number of frames
    if img.ndim != mask.ndim:
        raise ValueError("image and mask must have same ndim")

    if img.ndim == 2:
        return [img], [mask]

    image_frames, mask_frames = [], []
    np_images = []
    _, _, num_frames = img.GetLargestPossibleRegion().GetSize()

    for frame_num in range(num_frames):
        # Get the slices
        im_slice = extract_slice(img, frame_num, pix_type=IMAGE_PIXEL_TYPE)
        mask_slice = extract_slice(mask, frame_num, pix_type=MASK_PIXEL_TYPE)

        # Add
        image_frames.append(im_slice)
        mask_frames.append(mask_slice)

        # Add numpy array version of image to array so we can find
        # unique indices later
        np_im_slice = itk.array_from_image(im_slice)
        np_images.append(np_im_slice)

    # Find unique indices
    # TODO: round? This seems to be working fine!
    _, unique_indices = np.unique(np.array(np_images), axis=0, return_index=True)

    # Preserve order as much as possible. Without doing this, the unique_indices
    # are in order of the darkest frames first, increasingly getting brighter.
    unique_indices = np.sort(unique_indices)

    image_frames = [image_frames[i] for i in unique_indices]
    mask_frames = [mask_frames[i] for i in unique_indices]
    return image_frames, mask_frames

def split_filenames(xpaths, ypaths, train_split=0.6, val_split=0.2, shuffle=False):
    """
    Sorts the x and y filenames into train, val, and testing sets

    Parameters
    ----------
    xpaths : list of strings or pathlib.Paths
        Paths to images to use for training, val, or test
    ypaths : list of strings or pathlib.Paths
        Paths to masks to use for training, val, or test
    train_split : float
        Fraction of data to use as training data.
    val_split : float
        Fraction of data to use as validation data.
        1 - (train_split + val_split) is used as the testing split
    shuffle : bool
        Shuffle `xpaths` and `ypaths` before dividing into groups

    Returns
    -------
    train_imgs : list of pathlib.Path
        Training image paths
    train_masks : list of pathlib.Path
        Corresponding training mask paths
    val_imgs : list of pathlib.Path
        Validation image paths
    val_masks : list of pathlib.Path
        Corresponding validation mask paths
    test_imgs : list of pathlib.Path
        Testing image paths
    test_masks : list of pathlib.Path
        Corresponding testing mask paths
    """
    if len(xpaths) != len(ypaths):
        raise ValueError("length of xpaths and ypaths must be the same")

    if train_split + val_split > 1:
        raise ValueError("train_split and val_split can't sum to more than 1")

    if shuffle:
        xpaths, ypaths = sklearn.utils.shuffle(xpaths, ypaths)

    train_stop = int(train_split * len(xpaths))
    val_stop = train_stop + int(val_split * len(xpaths))

    train_imgs, train_masks = xpaths[:train_stop], ypaths[:train_stop]
    val_imgs, val_masks = xpaths[train_stop:val_stop], ypaths[train_stop:val_stop]
    test_imgs, test_masks = xpaths[val_stop:], ypaths[val_stop:]

    return train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks


def save_single_frame(img, mask, num, dir_):
    """
    Saves a single frame from an image and mask

    Parameters
    ----------
    img : itk.Image[,2]
        Image frame to write.
    mask : itk.Image[,2]
        Mask frame to write.
    num : int
        Example number. For example, the 43rd training frame.
        Will be encoded in the filename
    dir_ : str or pathlib.Path
        Directory to save to

    Returns
    -------
    None
    """
    if img.ndim != 2:
        raise ValueError(f"image must be 2D, not {img.ndim}D")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, not {mask.ndim}D")

    dir_ = path.Path(dir_)
    img_fname = "img_{}.mha".format(num)
    mask_fname = "mask_{}.mha".format(num)
    itk.imwrite(img, str(dir_ / img_fname))
    itk.imwrite(mask, str(dir_ / mask_fname))

def _extract_group(img_fnames, mask_fnames, dir_, frame_range_to_source_map=None):
    """
    Extracts an entire group of images and masks. For example,
    all of the training images.

    Parameters
    ----------
    img_fnames : list of strings
        paths to images
    mask_fnames : list of strings
        paths to masks
    dir_ : str or pathlib.Path
        Directory to extract to
    frame_range_to_source_map : dict of range, str
        Maps a range of frames / extracted images back to their source filename

    Returns
    -------
    The total number of frames written
    """
    total_num_frames_written = 0
    for img_fname, mask_fname in zip(img_fnames, mask_fnames):
        old_num_frames_written = total_num_frames_written
        img = itk.imread(img_fname, IMAGE_PIXEL_TYPE)
        mask = itk.imread(mask_fname, MASK_PIXEL_TYPE)
        img_frames, mask_frames = image_and_mask3d_to_list_of_unique_frames(img, mask)
        assert len(img_frames) == len(mask_frames)
        for img_frame, mask_frame in zip(img_frames, mask_frames):
            save_single_frame(img_frame, mask_frame, total_num_frames_written, dir_)
            total_num_frames_written += 1
        range_for_vid = range(old_num_frames_written, total_num_frames_written)
        if frame_range_to_source_map is not None:
            frame_range_to_source_map[range_for_vid] = img_fname

        assert total_num_frames_written - old_num_frames_written == len(img_frames)


    return total_num_frames_written

# TODO: This can probably be improved
def extract_data(
    train_img_fnames,
    train_mask_fnames,
    val_img_fnames,
    val_mask_fnames,
    test_img_fnames,
    test_mask_fnames,
    root_dir,
    overwrite_dirs=False,
    map_images_to_source=True
):
    """
    Extracts the train, val, and test images and masks to their corresponding
    directories. Subdirectories created by this function are
    "train", "val", and "test". Throws an error if these subdirs are present,
    or overwrites if overwrite_dirs is true.
    Images / frames will be named img_{num}.mha. Likewise, the corresponding
    masks will be named mask_{num}.mha.

    Parameters
    ----------
    train_img_fnames : list of strings
        Images to go in the training subdir
    train_mask_fnames : list of strings
        Masks to go in the training subdir
    val_img_fnames : list of strings
        Images to go in the validation subdir
    val_mask_fnames : list of strings
        Masks to go in the validation subdir
    test_img_fnames : list of strings
        Images to go in the testing subdir
    test_mask_fnames : list of strings
        Masks to go in the testing subdir
    root_dir : pathlib.Path:
        Root directory to write extracted data and
        the dictionaries mapping each image number back to its source filename
    overwrite_dirs : bool
        Overwrite the train, val, and test subdirs if present
    map_images_to_source : bool
        If true, write out a dictionary for the train, val, and test dirs
        mapping the image number back to the source file name. Writes them out
        to root_dir.

    Returns
    -------
    None
    """
    if len(train_img_fnames) != len(train_mask_fnames):
        raise ValueError("train_img_fnames and train_mask_fnames should have same length")

    if len(val_img_fnames) != len(val_mask_fnames):
        raise ValueError("val_img_fnames and val_mask_fnames should have same length")

    if len(test_img_fnames) != len(test_mask_fnames):
        raise ValueError("test_img_fnames and test_mask_fnames should have same length")

    # Ensure directories dont exist, or clear if overwrite_dirs was specified.
    train_data_dir = root_dir / TRAIN_DATA_DIR_NAME
    val_data_dir = root_dir / VAL_DATA_DIR_NAME
    test_data_dir = root_dir / TEST_DATA_DIR_NAME
    for p in [train_data_dir, val_data_dir, test_data_dir]:
        if p.exists():
            if overwrite_dirs:
                shutil.rmtree(str(p))
            else:
                err_msg = f"{str(p)} already exists. Please either remove the" \
                " directory, or if using the CLI, you can specify --force"
                raise FileExistsError(err_msg)

        p.mkdir(parents=True)
    train_data_to_source, val_data_to_source, test_data_to_source = None, None, None
    if map_images_to_source:
        train_data_to_source, val_data_to_source, test_data_to_source = {}, {}, {}

    num_train_frames = _extract_group(train_img_fnames, train_mask_fnames, train_data_dir, train_data_to_source)
    num_val_frames = _extract_group(val_img_fnames, val_mask_fnames, val_data_dir, val_data_to_source)
    num_test_frames = _extract_group(test_img_fnames, test_mask_fnames, test_data_dir, test_data_to_source)

    total_num_frames = num_train_frames + num_val_frames + num_test_frames
    print(f"Fraction of frames for training: {num_train_frames / total_num_frames}")
    print(f"Fraction of frames for validation: {num_val_frames / total_num_frames}")
    print(f"Fraction of frames for testing: {num_test_frames / total_num_frames}")

    if map_images_to_source:
        with open(str(root_dir / "train_data_key.p"), "wb") as f:
            pickle.dump(train_data_to_source, f)

        with open(str(root_dir / "val_data_key.p"), "wb") as f:
            pickle.dump(val_data_to_source, f)

        with open(str(root_dir / "test_data_key.p"), "wb") as f:
            pickle.dump(test_data_to_source, f)


def _read_data_dir(data_dirs):
    """
    Returns a list of images and a list of masks read from `data_dirs`.
    Assumes files in the directories specified by `data_dirs` are of the
    form `img_{num}.mha` and `mask_{num}.mha`. Returns them in order,
    i.e., imgs[i] has mask masks[i]

    Parameters
    ----------
    data_dirs : list of strings or pathlib.Paths
        Paths to the directories to read.

    Returns
    ----------
    List of dictionaries of strings. The ith element has keys 'x' and 'y',
    corresponding to the paths to the image and corresponding segmentation,
    respectively.
    """
    imgs, masks = [], []
    for dir_ in data_dirs:
        if isinstance(dir_, str):
            dir_ = path.Path(dir_)
        imgs.extend(natsorted(glob(str(dir_ / "img_*.mha"))))
        masks.extend(natsorted(glob(str(dir_ / "mask_*.mha"))))

    err_msg = f"Error: found {len(imgs)} images and {len(masks)} masks"
    assert len(imgs) == len(masks), err_msg
    return [{'x': imgs[i], 'y': masks[i]} for i in range(len(imgs))]


def get_data_loader(data_dirs, transforms, shuffle=True, num_workers=0):
    """
    Creates a DataLoader from the images and masks at `data_dir`. Incorporates
    transforms passed as parameters.

    Parameters
    ----------
    data_dirs : list of strings
        Path to the directory to read
    xtransforms : monai.transforms.Transform
        Transforms to apply to input images
    ytransforms : monai.transforms.Transform
        Transforms to apply to mask images
    num_workers : int
        Num workers to use when loading files from disk

    Returns
    ----------
    torch.utils.data.DataLoader
    """
    datalist = _read_data_dir(data_dirs)
    dataset = Dataset(datalist, transforms)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, pin_memory=torch.cuda.is_available(), shuffle=shuffle, num_workers=num_workers
    )

    return loader


def load_model(path, encoder_name="resnet34"):
    """
    Loads the model with params at `path`. Assumes the same architecture
    as `get_model`

    Parameters
    ----------
    path : string or pathlib.Path
        Path to the model params

    Returns
    ----------
    torch.nn.Module
    """

    model = get_model(encoder_name=encoder_name)
    model.load_state_dict(torch.load(str(path)))
    return model


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    model_name,
    model_save_dir,
    num_epochs,
    loss_function=None,
    monitor_with_tb=False,
    tb_logdir=None,
    verbose=True,
    status_logfile=None,
    test_loader=None,
):
    """
    Train a model

    Parameters
    ----------
    model : torch.nn.Module
    train_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    device : torch.device
    model_name : str
        Name of the best model decided by validation score
    model_name : pathlib.Path
        Directory to save the best model to
    num_epochs : int
    loss_function : torch.nn.modules.loss. Defaults to CELoss
    monitor_with_tb : bool
        Monitor the training loss and validation accuracy with tensorboard
    tb_logdir : str
        Logging directory for any tensorboard output.
        See tensorboard CLI logdir arg.

    Returns
    ----------
    List of validation metric scores (dice) per epoch
    """
    if monitor_with_tb:
        writer = SummaryWriter(tb_logdir)

    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()

    if status_logfile is not None:
        status_logfile = str(status_logfile)

    run_through_test_set = test_loader is not None

    model.to(device)
    # TODO: Keep track of the cumulative loss.
    optimizer = optim.Adam(model.parameters())
    metric_values = []
    iter_losses = []
    batch_sizes = []
    epoch_loss_values = []

    steps_per_epoch = len(train_loader.dataset) // train_loader.batch_size
    if len(train_loader.dataset) % train_loader.batch_size != 0:
        steps_per_epoch += 1

    def trans_batch(x, slice_=None, convert_label_to_one_hot=False, convert_pred_to_mask=True, convert_label_to_indices=False):
        if convert_label_to_indices and convert_label_to_one_hot:
            raise ValueError("convert_label_to_one_hot and convert_label_to_indices cannot both be True")
        preds = []
        labels = []
        for d in x:
            label = d["label"]
            pred = d["pred"]
            if convert_pred_to_mask:
                pred = output_to_3d_mask_transforms(pred)

            if convert_label_to_one_hot:
                t = Compose([AddChannel(), AsDiscrete(to_onehot=True, n_classes=3)])
                label = t(label)
            elif convert_label_to_indices:
                t = Compose([AsDiscrete(to_onehot=False, argmax=True, n_classes=3)])
                label = t(label).squeeze(0).to(dtype=torch.long)

            if slice_ is not None:
                pred = pred[slice_, :].unsqueeze(0)
                label = label[slice_, :].unsqueeze(0)

            preds.append(pred)
            labels.append(label)

        return preds, labels

    def prepare_batch_train(batchdata, device, non_blocking):
        return batchdata["x"].to(device), batchdata["y"].squeeze(1).to(device, dtype=torch.long)

    def prepare_batch_val(batchdata, device, non_blocking):
        return batchdata["x"].to(device), batchdata["y"].to(device)


    # We specify a key metric for the training, val, and test, even though
    # we only use the validation key metric to decide which models to save.
    # This is because none of the other metrics
    # (additional_metrics in the engines) will be run unless a key_metric
    # is specified
    key_train_metric = {
        "total_dice": MeanDice(
            include_background=False, output_transform=lambda x: trans_batch(x, convert_label_to_one_hot=True)
        ),
    }

    additional_train_metrics = {
        "eye_dice": MeanDice(
            include_background=True, output_transform=lambda x: trans_batch(x, 1, True)
        ),
        "nerve_dice": MeanDice(
            include_background=True, output_transform=lambda x: trans_batch(x, 2, True)
        ),
    }

    key_test_metric = {
        "total_dice": MeanDice(
            include_background=False, output_transform=trans_batch
        ),
    }

    additional_test_metrics = {
        "eye_dice": MeanDice(
            include_background=True, output_transform=lambda x: trans_batch(x, 1)
        ),
        "nerve_dice": MeanDice(
            include_background=True, output_transform=lambda x: trans_batch(x, 2)
        ),
        # TODO: Note that the loss here is unweighted.
        "loss": Loss(
            loss_function,
            output_transform=lambda x: trans_batch(x, convert_pred_to_mask=False, convert_label_to_indices=True)
        )
    }


    key_val_metric = {
        "total_dice": MeanDice(
            include_background=False, output_transform=trans_batch
        )
    }

    # TODO: Make these parameters to the function?
    additional_val_metrics = {
        "eye_dice": MeanDice(
            include_background=True, output_transform=lambda x: trans_batch(x, 1)
        ),
        "nerve_dice": MeanDice(
            include_background=True, output_transform=lambda x: trans_batch(x, 2)
        ),
        "loss": Loss(
            loss_function,
            output_transform=lambda x: trans_batch(x, convert_pred_to_mask=False, convert_label_to_indices=True))
    }


    model_save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_saver = CheckpointSaver(
        save_dir=model_save_dir,
        save_dict={"model": model},
        save_key_metric=True,
        key_metric_filename=model_name,
    )

    val_evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=model,
        key_val_metric=key_val_metric,
        additional_metrics=additional_val_metrics,
        prepare_batch=prepare_batch_val,
        val_handlers=[checkpoint_saver],
    )

    train_handlers = [ValidationHandler(1, val_evaluator)]
    if run_through_test_set:
        # The parameter "val_data_loader" may be confusing here
        # we're just checking the performance of the model on the testing
        # set after each epoch, only for informative purposes, not model selection
        test_evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=test_loader,
            network=model,
            key_val_metric=key_test_metric,
            additional_metrics=additional_test_metrics,
            prepare_batch=prepare_batch_val,
        )

        train_handlers.append(ValidationHandler(1, test_evaluator))

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=num_epochs,
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        key_train_metric=key_train_metric,
        additional_metrics=additional_train_metrics,
        loss_function=loss_function,
        train_handlers=train_handlers,
        prepare_batch=prepare_batch_train,
    )

    def write_images(engine, stage, num_images_to_save=5, max_alpha=.6):
        fig, axs = plt.subplots(5, num_images_to_save, constrained_layout=True)
        fig.set_size_inches(12, 8)
        for i in range(num_images_to_save):
            image_i = engine.state.output[i]["image"]
            pred_i = engine.state.output[i]["pred"] # 3 channel probabilities
            pred_i_one_hot = output_to_3d_mask_transforms(pred_i).unsqueeze(0)

            # This is one-hot if validation or testing. Otherwise, 1 channel
            # values 0-2.
            label_i_one_hot = engine.state.output[i]["label"]
            if stage == "Training":
                t = Compose([AddChannel(), AsDiscrete(to_onehot=True, n_classes=3)])
                label_i_one_hot = t(label_i_one_hot)
            label_i_one_hot = label_i_one_hot.unsqueeze(0)

            t = Compose([AsDiscrete(to_onehot=False, argmax=True, n_classes=3)])
            label_i_class_indices = t(label_i_one_hot.squeeze(0)).to(dtype=torch.long)
            pred_i_class_indices = t(pred_i_one_hot.squeeze(0))

            heatmap = output_to_3d_heatmap_transforms(pred_i)

            cmap = plt.get_cmap("jet")

            eye_heatmap = heatmap[1, :].cpu().squeeze(0).numpy()
            nerve_heatmap = heatmap[2, :].cpu().squeeze(0).numpy()

            eye_heatmap_with_colormap = cmap(eye_heatmap)
            nerve_heatmap_with_colormap = cmap(nerve_heatmap)

            eye_heatmap_with_colormap[:, :, 3] = max_alpha
            nerve_heatmap_with_colormap[:, :, 3] = max_alpha

            axs[0, i].imshow(image_i.cpu().squeeze(0), cmap="gray")
            axs[1, i].imshow(image_i.cpu().squeeze(0), cmap="gray")
            axs[1, i].imshow(eye_heatmap_with_colormap, vmin=0, vmax=1)
            axs[2, i].imshow(image_i.cpu().squeeze(0), cmap="gray")
            axs[2, i].imshow(nerve_heatmap_with_colormap, vmin=0, vmax=1)
            axs[3, i].imshow(pred_i_class_indices.squeeze(0).cpu())
            axs[4, i].imshow(label_i_class_indices.squeeze(0).cpu())


            raw_dice = compute_meandice(pred_i_one_hot, label_i_one_hot, include_background=False).cpu().squeeze(0)
            loss_amt = loss_function(pred_i.unsqueeze(0), label_i_class_indices).cpu()
            total_dice_score = np.nanmean(raw_dice)
            eye_dice_score = raw_dice[0]
            nerve_dice_score = raw_dice[1]

            fname = engine.state.batch[i]['x_meta_dict']['filename_or_obj']
            fname = path.Path(fname).name
            axs[0, i].set_title(fname)
            axs[1, i].set_title(f"Eye Dice = {eye_dice_score:.3f}")
            axs[2, i].set_title(f"Nerve Dice = {nerve_dice_score:.3f}")
            axs[3, i].set_title(f"Full Prediction\nloss = {loss_amt:.3f}, Dice = {total_dice_score:.3f}", wrap=True)
            axs[4, i].set_title("Ground Truth")

            for ax in axs[:, i]:
                ax.axis("off")

        last_col = num_images_to_save - 1
        cax1 = axs[1, last_col].inset_axes([1.05, 0, 0.05, 1], transform=axs[1, last_col].transAxes)
        cax2 = axs[2, last_col].inset_axes([1.05, 0, 0.05, 1], transform=axs[2, last_col].transAxes)
        fig.colorbar(cm.ScalarMappable(cmap=plt.get_cmap("jet")), cax=cax1)
        fig.colorbar(cm.ScalarMappable(cmap=plt.get_cmap("jet")), cax=cax2)


        # Now write images
        writer.add_figure(stage + "/img_samples", fig, engine.state.epoch)

    if monitor_with_tb:
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(event_filter=lambda engine, event: event % steps_per_epoch == len(train_loader) // 2),
            lambda x: write_images(x, "Training")
        )
        val_evaluator.add_event_handler(
            Events.ITERATION_COMPLETED(once=len(val_loader) // 2),
            lambda x: write_images(x, "Validation")
        )
        if run_through_test_set:
            test_evaluator.add_event_handler(
                Events.ITERATION_COMPLETED(once=len(test_loader) // 2),
                lambda x: write_images(x, "Testing")
            )

    @trainer.on(Events.ITERATION_COMPLETED)
    def _end_iter(engine):
        loss = np.average([o["loss"] for o in engine.state.output])
        iter_losses.append(loss)
        epoch = engine.state.epoch
        epoch_len = engine.state.max_epochs
        step = (engine.state.iteration % steps_per_epoch) + 1
        batch_len = len(engine.state.batch)
        batch_sizes.append(batch_len)

        if step % 100 == 0 and verbose:
            s = f"epoch {epoch}/{epoch_len}, step {step}/{steps_per_epoch}, training_loss = {loss:.4f}"
            if status_logfile:
                with open(status_logfile, "a+") as fp:
                    fp.write(s + "\n")
            else:
                print(s)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        overall_average_loss = np.average(iter_losses, weights=batch_sizes)
        epoch_loss_values.append(overall_average_loss)

        # clear the contents of iter_losses and batch_sizes for the next epoch
        iter_losses.clear()
        batch_sizes.clear()

        # fetch and report the validation metrics
        metric_value = val_evaluator.state.metrics["total_dice"]
        metric_values.append(metric_value)
        if verbose:
            s = f"evaluation for epoch {engine.state.epoch},  Dice = {metric_value:.4f}"
            if status_logfile:
                with open(status_logfile, "a+") as fp:
                    fp.write(s + "\n")
            else:
                print(s)

        if monitor_with_tb:
            writer.add_scalar("Training/loss", overall_average_loss, engine.state.epoch)
            writer.add_scalar("Training/dice_score",  trainer.state.metrics["total_dice"], engine.state.epoch)
            writer.add_scalar("Training/eye_dice", trainer.state.metrics["eye_dice"], engine.state.epoch)
            writer.add_scalar("Training/nerve_dice", trainer.state.metrics["nerve_dice"], engine.state.epoch)

            writer.add_scalar("Validation/loss", val_evaluator.state.metrics["loss"], engine.state.epoch)
            writer.add_scalar("Validation/dice_score",  val_evaluator.state.metrics["total_dice"], engine.state.epoch)
            writer.add_scalar("Validation/eye_dice", val_evaluator.state.metrics["eye_dice"], engine.state.epoch)
            writer.add_scalar("Validation/nerve_dice", val_evaluator.state.metrics["nerve_dice"], engine.state.epoch)

            if run_through_test_set:
                writer.add_scalar("Testing/loss", test_evaluator.state.metrics["loss"], engine.state.epoch)
                writer.add_scalar("Testing/dice_score",  test_evaluator.state.metrics["total_dice"], engine.state.epoch)
                writer.add_scalar("Testing/eye_dice", test_evaluator.state.metrics["eye_dice"], engine.state.epoch)
                writer.add_scalar("Testing/nerve_dice", test_evaluator.state.metrics["nerve_dice"], engine.state.epoch)

    trainer.run()
    if monitor_with_tb:
        writer.close()

    return metric_values


def test_model(
    model,
    test_loader,
    device,
):
    """
    Test a model. Prints and returns the dice score.

    Parameters
    ----------
    model : torch.nn.Module
    test_loader : torch.utils.data.DataLoader
    device : torch.device

    Returns
    ----------
    The dice score
    """

    model.to(device)

    def trans_batch_test(x):
        preds = []
        labels = []
        for d in x:
            preds.append(output_to_3d_mask_transforms(d["pred"]))
            labels.append(d["label"])
        return preds, labels

    def prepare_batch_test(batchdata, device, non_blocking):
        return batchdata["x"].to(device), batchdata["y"].squeeze(1).to(device, dtype=torch.long)

    key_val_metric = {
        "Dice": MeanDice(
            include_background=False, output_transform=trans_batch_test
        )
    }

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=model,
        key_val_metric=key_val_metric,
        prepare_batch=prepare_batch_test,
    )

    evaluator.run()

    print(evaluator.state.metrics["Dice"])
    return evaluator.state.metrics["Dice"]

def display_image_truth_and_prediction(image=None, truth=None, prediction=None):
    """
    Displays any / all of an image, ground truth value, and
    a models prediction. Expects that image, truth, and prediction
    are completely ready to be displayed. Any transforms should be
    applied before this function.

    Parameters
    ----------
    image : np.ndarray, torch.Tensor, or itk.Image[,2]
        Image to display
    truth : np.ndarray, torch.Tensor, or itk.Image[,2]
        Ground truth image to display
    prediction : np.ndarray, torch.Tensor, or itk.Image[,2]
        Predicted mask to display

    Returns
    ----------
    None
    """

    num_ims_to_display = 3 - ((image is None) + (truth is None) + (prediction is None))
    curr_image_num = 1

    fig = plt.figure(figsize=(10, 10))

    if image is not None:
        # Display the image
        sp1 = fig.add_subplot(1, num_ims_to_display, curr_image_num)
        sp1.set_title("Image")
        plt.imshow(image)

        curr_image_num += 1

    if truth is not None:
        # Display the truth
        sp2 = fig.add_subplot(1, num_ims_to_display, curr_image_num)
        sp2.set_title("Ground Truth")
        plt.imshow(truth, cmap="gray", vmin=0, vmax=2)

        curr_image_num += 1

    if prediction is not None:
        # Display the corresponding mask patch
        sp3 = fig.add_subplot(1, num_ims_to_display, curr_image_num)
        sp3.set_title("Prediction")
        plt.imshow(prediction, cmap="gray", vmin=0, vmax=2)

        curr_image_num += 1

    plt.show()

def get_predicted_mask_single_frame(model, input_image, device=None):
    '''
    Accepts a model, input image, and device. Gets the segmentation output
    from the model with the input_image as input. Returns the segmentation
    as an itk image in the original image space / size.
    '''
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    # Generate predictions
    with torch.no_grad():
        predicted_mask_raw = (
            model(itk_image_to_model_input(input_image).to(device)).cpu().squeeze(0)
        )
    predicted_mask = output_to_1d_mask_transforms(predicted_mask_raw)

    w, h = itk.size(input_image)
    transform = Compose(
        [
            AddChannel(),
            Resize((h, w), mode="nearest"),
        ]
    )
    # TODO: predicted_mask is one of two different types (image and array)
    #       Should probably streamline this
    predicted_mask = transform(predicted_mask).squeeze(0)
    predicted_mask = predicted_mask.astype(np.uint8)
    return image_from_array(
        predicted_mask, input_image, ttype=itk.Image[itk.UC, 2]
    )

def _run_inference_single_frame(model, input_image, device, get_onsd=True, debug_obj=None, **kwargs):
    """
    Runs a trained model on one 2D image. If `get_onsd` is True,
    uses the model's prediction to estimate the onsd

    Parameters
    ----------
    model : torch.nn.Module
    input_image : itk.Image[itk.F, 2]
    device : torch.device
    get_onsd : bool
        Use the model prediction to estimate the onsd
    debug_obj : tbitk.ocularus.ONSDDebugInfo
        Debug object that stores intermediate result from running the inference
        and onsd estimation code. If this is not None, its fields
        will be populated with these intermediate results.

    **kwargs : dict, optional
        Extra arguments to be passed to `calculate_onsd_from_mask`.

    Returns
    ----------
    tbitk.ai.inference_result.InferenceResult2D
    """

    onsd, score = None, None

    predicted_mask = get_predicted_mask_single_frame(model, input_image, device)
    if debug_obj is not None:
        debug_obj.mask = predicted_mask
    predicted_mask_arr = itk.array_from_image(predicted_mask)
    contains_eye_and_nerve = EYE_PIXEL_VALUE in predicted_mask_arr and NERVE_PIXEL_VALUE in predicted_mask_arr
    if contains_eye_and_nerve and get_onsd:
        onsd, score = calculate_onsd_from_mask(input_image, predicted_mask, debug_obj=debug_obj, **kwargs)
        if score is not None:
            score = score.astype(np.float64)

    return InferenceResult2D(
            input_image, predicted_mask, onsd, score
        )


def calculate_weights(data_dir):
    """
    Calculates the weights to be used during training to combat class imbalance.

    Parameters
    ----------
    data_dir : pathlib.Path or str
        Path to training data

    Returns
    ----------
    np.ndarray[1]
        The weights to be applied to each class during training.
        the ith index corresponds to the weight of the ith class.
    """

    if isinstance(data_dir, str):
        data_dir = path.Path(data_dir)

    mask_fnames = glob(str(data_dir / "mask_*.mha"))

    num_background, num_eye, num_nerve = 0, 0, 0
    for fname in mask_fnames:
        arr = itk.array_from_image(itk.imread(fname))
        num_background += (arr == 0).sum()
        num_eye += (arr == EYE_PIXEL_VALUE).sum()
        num_nerve += (arr == NERVE_PIXEL_VALUE).sum()

    print("num_background:", num_background)
    print("num_eye:", num_eye)
    print("num_nerve:", num_nerve)

    # This will probably be the background
    num_pixels_most_common_class = max(num_background, num_eye)
    return num_pixels_most_common_class / np.array([num_background, num_eye, num_nerve])


def predict_image_and_show(
    model,
    image_full_path,
    device=None,
    mask_full_path=None,
    show_alg_steps=False
):
    """
    Use `model` to generate a predicted mask, then display.
    Will also show the ground truth mask if `mask_full_path` is
    supplied.

    Parameters
    ----------
    model : torch.nn.Module
        Model that generates a prediction
    image_full_path : str
        Path to input image
    device : torch.device
        Device to generate prediction on
    mask_full_path : str
        Path to ground truth mask
    show_alg_steps : bool
        Show the intermediate progress of the onsd estimation algorithm.

    Returns
    ----------
    None
    """

    image = itk.imread(image_full_path, IMAGE_PIXEL_TYPE)
    mask = None if mask_full_path is None else itk.imread(mask_full_path)

    res_2d = run_inference(model, image, device=device, show_alg_steps=show_alg_steps)

    print("predicted_onsd:", res_2d.onsd)

    # Display
    display_image_truth_and_prediction(image, mask, res_2d.mask)


def run_inference(model, input_, device=None, get_onsd=True, debug_objs=None, **kwargs):
    """
    Runs a trained model on an input. If `get_onsd` is True,
    uses the model's prediction to estimate the onsd

    Parameters
    ----------
    model : torch.nn.Module
    input_ : list of itk.Image[itk.F, 2], itk.Image[itk.F, 2], or itk.Image[itk.F, 3]
    device : torch.device
    get_onsd : bool
        Use the model prediction to estimate the onsd
    debug_objs : ONSDDebugInfoCollection, optional
        collection that will be populated with ONSDDebugInfo for each frame, if present
    **kwargs : dict, optional
        Extra arguments to be passed to `calculate_onsd_from_mask`.

    Returns
    ----------
    tbitk.ai.inference_result.InferenceResult
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    store_debug_objs = debug_objs is not None

    input_is_list = isinstance(input_, list)
    input_is_2d_itk_img = isinstance(input_, itk.Image) and itk_image_is_2d(input_)
    input_is_3d_itk_img = isinstance(input_, itk.Image) and itk_image_is_video(input_)

    if input_is_2d_itk_img:
        debug_obj = ONSDDebugInfo(input_image=input_) if store_debug_objs else None
        res_2d = _run_inference_single_frame(model, input_, device, get_onsd, debug_obj=debug_obj, **kwargs)
        if store_debug_objs:
            debug_objs.append(debug_obj)
        return res_2d

    elif not input_is_list and not input_is_3d_itk_img:
        raise TypeError(
            "Input argument must either be a 2d itk image, 3d itk image, or list of 2d itk images. Got {}".format(
                type(input_)
            )
        )

    frame_num_to_onsd = {} if get_onsd else None
    frame_num_to_score = {} if get_onsd else None

    mask = []

    num_frames = len(input_) if input_is_list else input_.GetLargestPossibleRegion().GetSize()[2]

    for frame_num in range(num_frames):
        # Get the corresponding frame
        frame_ = input_[frame_num] if input_is_list else extract_slice(input_, frame_num)

        # Run inference
        debug_obj = ONSDDebugInfo(frame_) if store_debug_objs else None
        res_2d = _run_inference_single_frame(model, frame_, device, get_onsd, debug_obj=debug_obj, **kwargs)
        if store_debug_objs:
            debug_objs.append(debug_obj)

        if res_2d.onsd is not None:
            frame_num_to_onsd[frame_num] = res_2d.onsd

        if res_2d.score is not None:
            frame_num_to_score[frame_num] = res_2d.score

        mask.append(res_2d.mask) if input_is_list else mask.append(itk.array_from_image(res_2d.mask))

    onsds, scores = [], []
    for frame_num in frame_num_to_onsd:
        onsds.append(frame_num_to_onsd[frame_num])
        scores.append(frame_num_to_score[frame_num])

    onsd_for_video = aggregate_onsds(onsds, scores)

    if debug_objs is not None:
        debug_objs.calculate_percentiles()

    if input_is_3d_itk_img:
        mask = image_from_array(np.array(mask), input_)

    constructor = InferenceResult3DSingleSource if input_is_3d_itk_img else InferenceResult3DMultiSource
    return constructor(
        input_,
        mask,
        onsd_for_video,
        onsd_history=frame_num_to_onsd,
        score_history=frame_num_to_score,
    )


def generate_prediction_animation(res_3d, animation_save_path, actual_onsd=None, fps=24):
    """
    Accepts a 3d inference result and generates an animation. Either displays
    the animation or saves it to a directory depending on the arguments.
    The animation consist of the source video alongside the generated mask.
    The aggregated onsd estimation is also plotted per frame. The actual onsd
    an be plotted as a horizontal line as well.

    Parameters
    ----------
    res_3d : tbitk.ai.inference_result.InferenceResult3D
    animation_save_path : str
        Path to save the animation to. If not supplied,
        the animation is displayed
    actual_onsd : float
        The actual onsd of the source video
    fps : int
        FPS of the final animation

    Returns
    ----------
    None
    """
    total_num_frames = len(res_3d)
    initial_n_cols, initial_n_rows = res_3d[0].source.GetLargestPossibleRegion().GetSize()
    total_time = total_num_frames / fps

    y = []
    onsds_so_far = []
    scores_so_far = []

    fig = plt.figure()

    gs0 = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    gs1 = gs0[0].subgridspec(1, 2)
    ax1 = fig.add_subplot(gs1[0])
    ax2 = fig.add_subplot(gs1[1])
    ax3 = fig.add_subplot(gs0[1])

    ax1.axis("off")
    ax1.set_title("Predicted Eye and Nerve")
    ax2.axis("off")
    ax2.set_title("Ultrasound Source")
    ax3.set_xlim(0, total_time)
    ax3.set_ylim(0, 10)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Predicted ONSD (mm)")
    ax3.set_title("ONSD Measurement\nover Time")

    initial_data = np.zeros((initial_n_rows, initial_n_cols))
    prediction_fig = ax1.imshow(initial_data, animated=True, vmin=BACKGROUND_PIXEL_VALUE, vmax=NERVE_PIXEL_VALUE)
    source_fig = ax2.imshow(initial_data, animated=True, vmin=0, vmax=1)
    (line,) = ax3.plot([], y)
    if actual_onsd is not None:
        ax3.axhline(y=actual_onsd, color="m")

    def init():
        prediction_fig.set_data(initial_data)
        source_fig.set_data(initial_data)
        line.set_data([], y)

    def animate(i):
        # TODO: This is a temporary fix until itkpocus bug is fixed and
        # we can rely on the image pixel values being in the range 0-1.
        source_arr = itk.array_from_image(res_3d[i].source)
        source_fig.set_data(source_arr / np.max(source_arr))

        prediction_fig.set_data(res_3d[i].mask)

        # Make a prediction?
        if i in res_3d.onsd_history:
            onsds_so_far.append(res_3d[i].onsd)
            scores_so_far.append(res_3d[i].score)
            y.append(aggregate_onsds(onsds_so_far, scores_so_far))
        elif i:
            y.append(y[-1])
        else:
            y.append(0)

        line.set_data(np.array(range(len(y))) / fps, y)

        return [prediction_fig, source_fig, line]

    plt.tight_layout()

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=total_num_frames, repeat=False, interval=1000/fps
    )

    if animation_save_path:
        anim.save(animation_save_path, fps=fps)

    # TODO: Figure out why this needs to be an else...
    # cant save and display
    else:
        plt.show()


def save_debug_plots_to_dir(debug_objs, root_dir):
    """
    Generate debug plots and save them out to a directory.
    Images are named in order
    (debug_obj_frame_0.png, debug_obj_frame_1.png), etc.

    Parameters
    ----------
    debug_objs : tbitk.ocularus.ONSDDebugInfoCollection or list of tbitk.ocularus.ONSDDebugInfo
        Debug objects to generate and save plots from
    root_dir : str or Pathlib
        Root directory to save the plots to.
    """
    if isinstance(root_dir, str):
        root_dir = path.Path(root_dir)
    root_dir.mkdir()
    for i, debug_obj in enumerate(debug_objs):
        fig = debug_obj.create_debug_obj_figure()
        fig.savefig(root_dir / f"debug_obj_frame_{i}.png")
        fig.clf()
        plt.close(fig)


def get_filepaths_matching_pattern(pattern, recursive=False):
    """
    Find all files matching globs in `pattern`

    Parameters
    ----------
    pattern : list of str
        The glob patterns to match when searching for files
    recursive : bool
        Match ** to 0 or more directories.
        See the `recursive` argument:
        https://docs.python.org/3/library/glob.html#glob.glob

    Returns
    -------
    set of str
    """
    filepaths = []
    for p in pattern:
        filepaths.extend(glob(p, recursive=recursive))

    return set(filepaths)
