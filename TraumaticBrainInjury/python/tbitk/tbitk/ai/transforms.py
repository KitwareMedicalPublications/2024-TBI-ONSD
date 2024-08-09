import numpy as np

from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    Compose,
    EnsureTyped,
    Resized,
    LoadImaged,
    RandFlipd,
    RandZoomd,
    ScaleIntensityd,
    Transform,
    EnsureType,
    Activations,
    AsDiscrete,
    SqueezeDim,
    KeepLargestConnectedComponent,
    AddChannel,
    Resize,
    ScaleIntensity,
)
from tbitk.ai.constants import NETWORK_INPUT_SHAPE

# TODO add more comments
# These only work for single images
class ConvertToSingleChannelBackgroundEyeNerve(Transform):
    """
    Convert multi channel mask to single channel.
    Input has the following labels in a 3d array:
        2d slice 0 is a binary mask for the background,
        2d slice 1 is a binary mask for the eye,
        2d slice 2 is a binary mask for the nerve,
    Convert this to a 2d array with:
        0's where the background is
        1's where the eye is
        2's where the nerve is
    """

    def __call__(self, data):
        num_masks, h, w = data.shape
        out = np.zeros((h, w))
        for i in range(1, num_masks):
            slice = data[i, :, :]
            out[slice == 1] = i
        return out

class UnSqueezeDim(Transform):
    def __init__(self, dim = 0):
        if dim is None:
            raise TypeError("dim must not be None")
        if not isinstance(dim, int):
            raise TypeError(f"dim must be an int but is {type(dim).__name__}.")
        self.dim = dim

    def __call__(self, img):
        # TODO: Add better checking here.
        return img.unsqueeze(self.dim)



# Define our transforms
train_transforms = Compose(
    [
        LoadImaged(keys=["x", "y"], image_only=False),
        EnsureTyped(keys=["x", "y"]),
        AddChanneld(keys=["x", "y"]),
        Resized(keys=["x", "y"], spatial_size=NETWORK_INPUT_SHAPE, mode="nearest"),
        # RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.3, mode="nearest"), # TODO: Make sure we dont center zoom at cut out the top. Uncomment
        ScaleIntensityd(keys=["x"]),
        RandFlipd(keys=["x", "y"], prob=0.5, spatial_axis=1),
    ]
)

eval_transforms = Compose(
    [
        LoadImaged(keys=["x", "y"], image_only=False),
        EnsureTyped(keys=["x", "y"]),
        AddChanneld(keys=["x", "y"]),
        Resized(keys=["x", "y"], spatial_size=NETWORK_INPUT_SHAPE, mode="nearest"),
        ScaleIntensityd(keys=["x"]),
        EnsureTyped(keys=["x", "y"]),
        AsDiscreted(keys=["y"], to_onehot=True, n_classes=3),
    ]
)

# TODO: keep largest connected component
# TODO: Incorporate below transform
output_to_3d_mask_transforms = Compose(
    [
        EnsureType(),
        Activations(softmax=True),
        AsDiscrete(argmax=True, to_onehot=True, n_classes=3),
    ]
)

output_to_3d_heatmap_transforms = Compose(
    [
        EnsureType(),
        Activations(softmax=True),
    ]
)

output_to_1d_mask_transforms = Compose(
    [
        EnsureType(),
        Activations(softmax=True),
        AsDiscrete(argmax=True, to_onehot=True, n_classes=3),
        ConvertToSingleChannelBackgroundEyeNerve(),
        EnsureType(),
        UnSqueezeDim(dim=0),
        KeepLargestConnectedComponent(applied_labels=(1, 2)),
        SqueezeDim(dim=0),
        EnsureType(data_type="numpy"),
    ]
)

# TODO: ensuretype doesnt do the conversion from itk to tensor
itk_image_to_model_input = Compose(
    [
        EnsureType(),
        AddChannel(),
        Resize(NETWORK_INPUT_SHAPE, mode="nearest"),
        ScaleIntensity(),
        AddChannel(),
        EnsureType(),
    ]
)