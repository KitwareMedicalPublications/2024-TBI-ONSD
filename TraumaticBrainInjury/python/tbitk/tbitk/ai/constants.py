import itk

# Define some helpful constants
NUM_CLASSES = 3  # background, eye, nerve
NETWORK_INPUT_SHAPE = (256, 256)
BATCH_SIZE = 16


DEFAULT_BEST_MODEL_NAME = "best_model.pt"

TRAIN_DATA_DIR_NAME = "train"
VAL_DATA_DIR_NAME = "val"
TEST_DATA_DIR_NAME = "test"

# ITK stuff
IMAGE_PIXEL_TYPE = itk.F
MASK_PIXEL_TYPE = itk.UC

IMAGE_TYPE = itk.Image[IMAGE_PIXEL_TYPE, 2]
VIDEO_TYPE = itk.Image[IMAGE_PIXEL_TYPE, 3]
MASK_TYPE = itk.Image[MASK_PIXEL_TYPE, 2]
VIDEO_MASK_TYPE = itk.Image[MASK_PIXEL_TYPE, 3]

# When showing the dice scores for each of the 3 classes, should
# we show the dice score for the background? Affects validation / model selection.
INCLUDE_BACKGROUND = False
DEFAULT_NUM_EPOCHS = 10

BACKGROUND_PIXEL_VALUE = 0
EYE_PIXEL_VALUE = 1
NERVE_PIXEL_VALUE = 2

AVAILABLE_LOSS_STRS = ["ce", "focal"]