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
from lesson_functions import draw_boxes


IMG_FILE="../test_images/test_image.jpg"
TEST_OUT_DIR="hotspot"

class TestPipeline(unittest.TestCase):
    
    def setUp(self):
        if not os.path.exists(TEST_OUT_DIR):
            os.makedirs(TEST_OUT_DIR)

    def test_hotspot(self) :
        # Read in a pickle file with bboxes saved
        # Each item in the "all_bboxes" list will contain a
        # list of boxes for one of the images shown above
        box_list = pickle.load( open( "bbox_pickle.p", "rb" ))

        # Read in image similar to one shown above
        image = mpimg.imread(IMG_FILE)
        image_with_boxes = draw_boxes(image,box_list)
        
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
 
        # Add heat to each box in box list
        heat = uut.add_heat(heat,box_list)

        # Apply threshold to help remove false positives
        heat = uut.apply_threshold(heat,1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = uut.draw_labeled_bboxes(np.copy(image), labels)
        
        final_image = image_util.arrangeImages([image, image_with_boxes, heatmap, draw_img],["Original", "Boxes that detected car", "Heatmap", "labeld boxes"], 2)
        image_util.saveImage(final_image, TEST_OUT_DIR+"/hotspot.jpg")



if __name__ == '__main__':
    unittest.main()
