import unittest
import tbitk.cvat as cvat
import os
import shutil

TMP_DIR = './tests/tmp'

class TestSingleInstanceVideoAnnotation(unittest.TestCase):
    def setUp(self):
        os.mkdir(TMP_DIR)
        
    def tearDown(self):
        shutil.rmtree(TMP_DIR)
    
    def test_parse(self):
        fp = './tests/data/task_capture_1_2021-02-03t13-47-33.mp4-2021_08_20_16_56_09-cvat for video 1.1.zip'
        xml_fp = cvat.extract_and_rename(fp, TMP_DIR)
        siva = cvat.SingleInstanceVideoAnnotation.parse(xml_fp)
        keys = siva.instances.keys()
        self.assertTrue('eye' in keys)
        self.assertTrue('nerve' in keys)
        self.assertTrue(len(keys) == 2)

if __name__ == '__main__':
    unittest.main()
