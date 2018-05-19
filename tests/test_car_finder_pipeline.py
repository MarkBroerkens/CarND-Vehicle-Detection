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
IMG_DIR2="../input_videos"
TEST_OUT_DIR="pipeline"

class TestCarFinderPipeline(unittest.TestCase):
    
    def setUp(self):
        if not os.path.exists(TEST_OUT_DIR):
            os.makedirs(TEST_OUT_DIR)

    def _test_process(self) :
        p = pipeline.Pipeline(1)
        imgs = image_util.loadImagesRGB(IMG_DIR)
        for i,img in enumerate(imgs):
            processed_img = p.process(np.copy(img))
            
            out = image_util.arrangeImages([img, processed_img], ["original","car detection"], figsize=(4,2))
            image_util.saveImage(out, TEST_OUT_DIR+"/identified_boxes"+str(i)+".png")

    def test_readme_video_images(self) :
        # images from sequence in video
        p = pipeline.Pipeline(1)
        imgs = image_util.loadImagesRGB(IMG_DIR2)
        for i,img in enumerate(imgs):
            draw_img, labels, heatmap, boxed_image = p.process_verbose(np.copy(img))
        
            out = image_util.arrangeImages([img, boxed_image, heatmap, labels[0], draw_img], ["original","car detections", "heatmap", "labels", "result"], figsize=(5,1))
            image_util.saveImage(out, TEST_OUT_DIR+"/readme_videoprocess"+str(i)+".png")
        p = pipeline.Pipeline(9)
        imgs = image_util.loadImagesRGB(IMG_DIR2)
        for i,img in enumerate(imgs):
            draw_img, labels, heatmap, boxed_image = p.process_verbose(np.copy(img))
            out = image_util.arrangeImages([img, boxed_image, heatmap, labels[0], draw_img], ["original","car detections", "heatmap", "labels", "result"], figsize=(5,1))
            image_util.saveImage(out, TEST_OUT_DIR+"/readme_videoprocess_with_history"+str(i)+".png")

    def test_readme_video_images(self) :
        # images from test image folder
        p = pipeline.Pipeline(1)
        imgs = image_util.loadImagesRGB(IMG_DIR)
        for i,img in enumerate(imgs):
            draw_img, labels, heatmap, boxed_image = p.process_verbose(np.copy(img))
            
            out = image_util.arrangeImages([img, boxed_image, heatmap, labels[0], draw_img], ["original","car detections", "heatmap", "labels", "result"], figsize=(5,1))
            image_util.saveImage(out, TEST_OUT_DIR+"/readme_test_images_process"+str(i)+".png")




if __name__ == '__main__':
    unittest.main()
