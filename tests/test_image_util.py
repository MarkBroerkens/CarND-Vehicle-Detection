import sys
import os
import unittest
import cv2

sys.path.append("..")
import image_util as uut

TEST_IMAGES_DIR="../test_images/"
TEST_OUT_DIR="test_image_util_out"

class ImageUtilTest(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(TEST_OUT_DIR):
            os.makedirs(TEST_OUT_DIR)
    
    def tearDown(self):
        return

    def test_01_findImageFilesFlat(self):
        image_files_list = uut.findImageFilesFlat(TEST_IMAGES_DIR)
        self.assertEqual(5,len(image_files_list))
    
    def test_01_findImageFilesDeep(self):
        image_files_list = uut.findImageFilesDeep(TEST_IMAGES_DIR)
        self.assertEqual(5,len(image_files_list))



if __name__ == '__main__':
    unittest.main()
