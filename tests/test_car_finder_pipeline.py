import matplotlib
matplotlib.use('Agg')
import unittest
import os
import sys
import numpy as np
import cv2

sys.path.append("..")
import car_finder_pipeline as pipeline
import image_util
import train


IMG_DIR="../test_images"
TEST_OUT_DIR="pipeline"

class TestCarFinderPipeline(unittest.TestCase):
    
    def setUp(self):
        if not os.path.exists(TEST_OUT_DIR):
            os.makedirs(TEST_OUT_DIR)

    def test_process(self) :
        p = pipeline.Pipeline()
        imgs = image_util.loadImagesRGB(IMG_DIR)
        for i,img in enumerate(imgs):
            #img = img.astype(np.float32)/255
            processed_img = p.process(img)
            
            image_util.saveImage(processed_img, TEST_OUT_DIR+"/identified_boxes"+str(i)+".jpg")

if __name__ == '__main__':
    unittest.main()
