import itk
import unittest
import tbitk.ai.dl_cli as dl_cli
from pathlib import Path
import tempfile
import pickle
import numpy as np
import tbitk.data_manager as dm


class ExtractTestCase(unittest.TestCase):
    WORKING_DIR = Path(__file__).parents[0]
    AI_TEST_DIR = Path(__file__).parents[1]

    A1_VIDEO_PATH = (AI_TEST_DIR / "data/preprocessed/a-1.mha").resolve()
    A2_VIDEO_PATH = (AI_TEST_DIR / "data/preprocessed/a-2.mha").resolve()
    B2_VIDEO_PATH = (AI_TEST_DIR / "data/preprocessed/b-2.mha").resolve()
    E1_VIDEO_PATH = (AI_TEST_DIR / "data/preprocessed/e-1.mha").resolve()

    VIDEO_PATHS = [A1_VIDEO_PATH, A2_VIDEO_PATH, B2_VIDEO_PATH, E1_VIDEO_PATH]

    EXTRACT_SUBDIR = "extract_root"

    def _write_small_videos(self, twritedir):
        twritedir = Path(twritedir)
        # Make a preprocessed and and annotation directories
        preprocessed_dir = twritedir / "preprocessed"
        annotation_dir = twritedir / "annotation"
        raw_dir = twritedir / "raw"

        preprocessed_dir.mkdir()
        annotation_dir.mkdir()
        raw_dir.mkdir()

        # All 3 frame, 2x2 videos filled with
        # pixel values == 2, 3, 4, and 5, respectively.
        VIDEO_SIZE = (3, 2, 2)

        vid_1 = itk.image_from_array(np.full(VIDEO_SIZE, 2, dtype=np.single), ttype=itk.Image[itk.F, 3])
        vid_2 = itk.image_from_array(np.full(VIDEO_SIZE, 3, dtype=np.single), ttype=itk.Image[itk.F, 3])
        vid_3 = itk.image_from_array(np.full(VIDEO_SIZE, 4, dtype=np.single), ttype=itk.Image[itk.F, 3])
        vid_4 = itk.image_from_array(np.full(VIDEO_SIZE, 5, dtype=np.single), ttype=itk.Image[itk.F, 3])

        mask_1 = itk.image_from_array(np.random.randint(2, size=VIDEO_SIZE, dtype=np.ubyte), ttype=itk.Image[itk.UC, 3])
        mask_2 = itk.image_from_array(np.random.randint(2, size=VIDEO_SIZE, dtype=np.ubyte), ttype=itk.Image[itk.UC, 3])
        mask_3 = itk.image_from_array(np.random.randint(2, size=VIDEO_SIZE, dtype=np.ubyte), ttype=itk.Image[itk.UC, 3])
        mask_4 = itk.image_from_array(np.random.randint(2, size=VIDEO_SIZE, dtype=np.ubyte), ttype=itk.Image[itk.UC, 3])

        vid_1_preproc_path = preprocessed_dir / "vid_1.mha"
        vid_2_preproc_path = preprocessed_dir / "vid_2.mha"
        vid_3_preproc_path = preprocessed_dir / "vid_3.mha"
        vid_4_preproc_path = preprocessed_dir / "vid_4.mha"

        # Preprocessed and raw are exactly the same.
        # This is just to trick the data manager
        vid_1_raw_path = raw_dir / "vid_1.mha"
        vid_2_raw_path = raw_dir / "vid_2.mha"
        vid_3_raw_path = raw_dir / "vid_3.mha"
        vid_4_raw_path = raw_dir / "vid_4.mha"

        mask_1_path = annotation_dir / "vid_1-label.mha"
        mask_2_path = annotation_dir / "vid_2-label.mha"
        mask_3_path = annotation_dir / "vid_3-label.mha"
        mask_4_path = annotation_dir / "vid_4-label.mha"

        itk.imwrite(vid_1, str(vid_1_preproc_path))
        itk.imwrite(vid_2, str(vid_2_preproc_path))
        itk.imwrite(vid_3, str(vid_3_preproc_path))
        itk.imwrite(vid_4, str(vid_4_preproc_path))

        itk.imwrite(mask_1, str(mask_1_path))
        itk.imwrite(mask_2, str(mask_2_path))
        itk.imwrite(mask_3, str(mask_3_path))
        itk.imwrite(mask_4, str(mask_4_path))

        itk.imwrite(vid_1, str(vid_1_raw_path))
        itk.imwrite(vid_2, str(vid_2_raw_path))
        itk.imwrite(vid_3, str(vid_3_raw_path))
        itk.imwrite(vid_4, str(vid_4_raw_path))

        return vid_1_preproc_path, vid_2_preproc_path, vid_3_preproc_path, vid_4_preproc_path

    def _load_keys(self, extract_root):
        with open(str(extract_root / "train_data_key.p"), "rb") as f:
            train_key = pickle.load(f)
        with open(str(extract_root / "val_data_key.p"), "rb") as f:
            val_key = pickle.load(f)
        with open(str(extract_root / "test_data_key.p"), "rb") as f:
            test_key = pickle.load(f)

        return train_key, val_key, test_key

    def _check_no_overlap(self, train_key, val_key, test_key):
        self.assertEqual(len(set(train_key.values()) & set(val_key.values())), 0)
        self.assertEqual(len(set(train_key.values()) & set(test_key.values())), 0)
        self.assertEqual(len(set(val_key.values()) & set(test_key.values())), 0)

    def _check_subdir_num_frames(self, map_, dir_):
        total_num_frames = 0
        for r, fp in map_.items():
            im = itk.imread(str(fp))
            arr = itk.array_from_image(im)
            num_unique_frames = len(np.unique(arr, axis=0))
            total_num_frames += num_unique_frames
            self.assertEqual(r.stop - r.start, num_unique_frames)

        # Note the extra factor of 2 here for the mask frames
        self.assertEqual(2 * total_num_frames, len(list(dir_.glob("*.mha"))))

    def _check_frames(self, map_, dir_):
        dir_ = Path(dir_)
        for r, fp in map_.items():
            img = itk.imread(str(fp))
            img_arr = itk.array_from_image(img)
            mask = itk.imread(dm.get_filepaths(str(fp))["annotation_label_image"])
            mask_arr = itk.array_from_image(mask)
            for i in r:
                img_frame = itk.imread(str(dir_ / f"img_{i}.mha"))
                mask_frame = itk.imread(str(dir_ / f"mask_{i}.mha"))

                img_frame_arr = itk.array_from_image(img_frame)
                mask_frame_arr = itk.array_from_image(mask_frame)

                self.assertIn(img_frame_arr, img_arr)
                self.assertIn(mask_frame_arr, mask_arr)


class ExtractSpecifiedDirsTestCase(ExtractTestCase):
    def test_basic(self):
        for vp in self.VIDEO_PATHS:
            self.assertTrue(vp.exists())

        with tempfile.TemporaryDirectory() as tdir:
            EXTRACT_PATH = Path(tdir) / self.EXTRACT_SUBDIR
            self.assertFalse(EXTRACT_PATH.exists())

             # Now construct cli args
            args = [
                "extract",
                "--root_dir", str(EXTRACT_PATH),
                "--train_patterns", str(self.A1_VIDEO_PATH), str(self.A2_VIDEO_PATH),
                "--val_patterns", str(self.B2_VIDEO_PATH),
                "--test_patterns", str(self.E1_VIDEO_PATH),
            ]

            # Run CLI
            dl_cli.main(args)

             # Verify
            self.assertTrue(EXTRACT_PATH.exists())
            for subdir_name in ["train", "val", "test"]:
                subdir_path = EXTRACT_PATH / subdir_name
                self.assertTrue(subdir_path.exists())

            train_key, val_key, test_key = self._load_keys(EXTRACT_PATH)
            print(train_key)
            print(val_key)
            print(test_key)

            # Check sizes of the maps
            self.assertEqual(len(train_key.values()), 2)
            self.assertEqual(len(val_key.values()), 1)
            self.assertEqual(len(test_key.values()), 1)

            # Now check the content.
            # Train key
            self.assertIn(str(self.A1_VIDEO_PATH), train_key.values())
            self.assertIn(str(self.A2_VIDEO_PATH), train_key.values())
            self.assertNotIn(str(self.B2_VIDEO_PATH), train_key.values())
            self.assertNotIn(str(self.E1_VIDEO_PATH), train_key.values())

            # Val key
            self.assertNotIn(str(self.A1_VIDEO_PATH), val_key.values())
            self.assertNotIn(str(self.A2_VIDEO_PATH), val_key.values())
            self.assertIn(str(self.B2_VIDEO_PATH), val_key.values())
            self.assertNotIn(str(self.E1_VIDEO_PATH), val_key.values())

            # Test key
            self.assertNotIn(str(self.A1_VIDEO_PATH), test_key.values())
            self.assertNotIn(str(self.A2_VIDEO_PATH), test_key.values())
            self.assertNotIn(str(self.B2_VIDEO_PATH), test_key.values())
            self.assertIn(str(self.E1_VIDEO_PATH), test_key.values())

            # Now check that the number of frames written out match the expected
            for map_, subdir in zip([train_key, val_key, test_key], ["train", "val", "test"]):
                self._check_subdir_num_frames(map_, EXTRACT_PATH / subdir)

    def test_frames_extracted(self):
        # Write them out
        with tempfile.TemporaryDirectory() as twritedir, tempfile.TemporaryDirectory() as textractdir:
            vid_1_path, vid_2_path, vid_3_path, vid_4_path = self._write_small_videos(twritedir)

            EXTRACT_PATH = Path(textractdir) / self.EXTRACT_SUBDIR

            # Now construct cli args
            args = [
                "extract",
                "--root_dir", str(EXTRACT_PATH),
                "--train_patterns", str(vid_1_path), str(vid_2_path),
                "--val_patterns", str(vid_3_path),
                "--test_patterns", str(vid_4_path),
            ]

            # Run CLI
            dl_cli.main(args)

            train_key, val_key, test_key = self._load_keys(EXTRACT_PATH)
            self._check_frames(train_key, EXTRACT_PATH / "train")
            self._check_frames(val_key, EXTRACT_PATH / "val")
            self._check_frames(test_key, EXTRACT_PATH / "test")

class ExtractFilePatternsTestCase(ExtractTestCase):
    def test_basic(self):
        for vp in self.VIDEO_PATHS:
            self.assertTrue(vp.exists())
        with tempfile.TemporaryDirectory() as tdir:
            EXTRACT_PATH = Path(tdir) / self.EXTRACT_SUBDIR
            self.assertFalse(EXTRACT_PATH.exists())

            # Now construct cli args
            args = [
                "extract",
                "--root_dir", str(EXTRACT_PATH),
                "--file_patterns",
            ]
            for vp in self.VIDEO_PATHS:
                args.append(str(vp))

            args.extend([
                "--train_split", "0.6",
                "--val_split", "0.25"
            ])

            # Run CLI
            dl_cli.main(args)

            # Verify
            self.assertTrue(EXTRACT_PATH.exists())
            for subdir_name in ["train", "val", "test"]:
                subdir_path = EXTRACT_PATH / subdir_name
                self.assertTrue(subdir_path.exists())

            train_key, val_key, test_key = self._load_keys(EXTRACT_PATH)

            print(train_key)
            print(val_key)
            print(test_key)

            # Check that all of the videos were written out
            num_videos_written = len(set(train_key.values()) | set(val_key.values()) | set(test_key.values()))
            self.assertEqual(num_videos_written, len(self.VIDEO_PATHS))
            self.assertEqual(len(train_key.values()), 2)
            self.assertEqual(len(val_key.values()), 1)
            self.assertEqual(len(test_key.values()), 1)

            # Check that each video appears in only one of the three sets
            self._check_no_overlap(train_key, val_key, test_key)

            # Now check that the number of frames written out match the expected
            for map_, subdir in zip([train_key, val_key, test_key], ["train", "val", "test"]):
                self._check_subdir_num_frames(map_, EXTRACT_PATH / subdir)

    def test_frames_extracted(self):
        with tempfile.TemporaryDirectory() as twritedir, tempfile.TemporaryDirectory() as textractdir:
            vid1_path, vid2_path, vid3_path, vid4_path = self._write_small_videos(twritedir)

            EXTRACT_PATH = Path(textractdir) / self.EXTRACT_SUBDIR

            # Now construct cli args
            args = [
                "extract",
                "--root_dir", str(EXTRACT_PATH),
                "--file_patterns",
                str(vid1_path), str(vid2_path),
                str(vid3_path), str(vid4_path),
                "--train_split", "0.5",
                "--val_split", "0.25"
            ]

            # Run CLI
            dl_cli.main(args)

            train_key, val_key, test_key = self._load_keys(EXTRACT_PATH)
            self._check_frames(train_key, EXTRACT_PATH / "train")
            self._check_frames(val_key, EXTRACT_PATH / "val")
            self._check_frames(test_key, EXTRACT_PATH / "test")


if __name__ == "__main__":
    unittest.main()
