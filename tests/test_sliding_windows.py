import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import os
import sys
import numpy as np
import cv2
import pickle
from scipy.ndimage.measurements import label


sys.path.append("..")
import hotspots as uut
import image_util
from lesson_functions import draw_boxes, slide_window


IMG_FILE_DIR="../test_images"
TEST_OUT_DIR="slinding_windows/"

class TestPipeline(unittest.TestCase):
    
    def setUp(self):
        if not os.path.exists(TEST_OUT_DIR):
            os.makedirs(TEST_OUT_DIR)

    def draw_sliding_windows(self, filename, img, y_start, y_stop, scale):
        xy_window = (int(64*scale),int(64*scale))
        boxes = slide_window(img, y_start_stop=[y_start, y_stop], xy_window=xy_window, xy_overlap=(0.75, 0.75))
        img = draw_boxes(img,boxes)
        img = cv2.rectangle(img,(0,y_start),(xy_window[0],y_start+xy_window[1]), (255,0,0), 6)
        print (len(boxes))
        print (img.shape)
        image_util.saveImage(img, filename)
    
    
    

    def test_sliding_windows_1(self) :
        imgs = image_util.loadImagesRGB(IMG_FILE_DIR)
        for img in imgs:
            # search with box size 64 * 1 = 64
            ystart = 380
            ystop = 480
            scale = 1.0
            self.draw_sliding_windows(TEST_OUT_DIR+"windows_1.png", img, ystart, ystop, scale)
            
            # search with box size 64 * 1.5 = 96
            ystart = 400
            ystop = 600
            scale = 1.5
            self.draw_sliding_windows(TEST_OUT_DIR+"windows_2.png",img,  ystart, ystop, scale)
            
            
            # search with box size 64 * 2.5 = 160
            ystart = 500
            ystop = 700
            scale = 2.5
            self.draw_sliding_windows(TEST_OUT_DIR+"windows_3.png", img, ystart, ystop, scale)



if __name__ == '__main__':
    unittest.main()
