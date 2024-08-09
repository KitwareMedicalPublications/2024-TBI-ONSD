import unittest
import itk
import tbitk.ai.deep_learning as dl

from monai.metrics import compute_meandice
from monai.transforms import AddChannel, AsDiscrete, Compose, EnsureType
from pathlib import Path

class TestSimpleImageInference(unittest.TestCase):
    ai_test_dir = Path(__file__).parents[2]
    # TODO: Use different images? No cvat_raw or raw
    MODEL_PATH = str((ai_test_dir / "models/example_model.pt").resolve())
    IMAGE_PATH = str((ai_test_dir / "data/preprocessed/img.mha").resolve())
    GT_MASK_PATH = str((ai_test_dir / "data/annotation/img-label.mha").resolve())

    def test_model_inference_dice(self):
        # Load
        model = dl.load_model(path=self.MODEL_PATH)
        image = itk.imread(self.IMAGE_PATH)
        gt_mask = itk.imread(self.GT_MASK_PATH)

        # Predict
        inf_res = dl.run_inference(model, image)
        predicted_mask = inf_res.mask

        # Preprocess
        transform = Compose(
            [
                AddChannel(), #[HxW] -> [1xHxW]
                EnsureType(),
                AsDiscrete(to_onehot=True, n_classes=3), #[1xHxW] -> [3xHxW]
                AddChannel() #[3xHxW] -> [1x3xHxW]
            ]
        )

        gt_mask_eval = transform(gt_mask)
        predicted_mask_eval = transform(predicted_mask)

        # Evaluate
        dice = compute_meandice(predicted_mask_eval, gt_mask_eval).squeeze()
        dice_background = dice[0].item()
        dice_eye = dice[1].item()
        dice_nerve = dice[2].item()

        self.assertGreaterEqual(dice_background, 0.9)
        self.assertGreaterEqual(dice_nerve, 0.9)
        self.assertGreaterEqual(dice_eye, 0.9)

if __name__ == "__main__":
    unittest.main()
