import unittest
import tbitk.ai.deep_learning as dl


class TestChangingEncoder(unittest.TestCase):
    encoder_names = ["resnet34", "timm-regnety_080"]
    unet_name_format = "u-{}"

    def test_encoder_names(self):
        for encoder_name in self.encoder_names:
            model = dl.get_model(encoder_name=encoder_name)
            self.assertEqual(model.name, self.unet_name_format.format(encoder_name))
