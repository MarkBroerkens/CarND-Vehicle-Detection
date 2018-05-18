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
        p = pipeline.Pipeline(1)
        imgs = image_util.loadImagesRGB(IMG_DIR)
        for i,img in enumerate(imgs):
            #img = img.astype(np.float32)/255
            processed_img = p.process(np.copy(img))
            
            out = image_util.arrangeImages([img, processed_img], ["original","car detection"], figsize=(4,2))
            image_util.saveImage(out, TEST_OUT_DIR+"/identified_boxes"+str(i)+".png")

    def test_readme_images(self) :
        p = pipeline.Pipeline(1)
        imgs = image_util.loadImagesRGB(IMG_DIR)
        for i,img in enumerate(imgs):
            #img = img.astype(np.float32)/255
            draw_img, heatmap, boxed_image = p.process_verbose(np.copy(img))
        
            out = image_util.arrangeImages([img, boxed_image, heatmap, draw_img], ["original","car detections", "heatmap", "result"], figsize=(4,1))
            image_util.saveImage(out, TEST_OUT_DIR+"/identified_boxes"+str(i)+".png")


if __name__ == '__main__':
    unittest.main()
