import sys
import os
import unittest
import cv2
import numpy as np


sys.path.append("..")
import lesson_functions as uut
import image_util

LABELED_DATA_VEHICLE_DIR="../labeled_data/vehicles"
LABELED_DATA_NON_VEHICLE_DIR="../labeled_data/non-vehicles"
TEST_OUT_DIR="test_features_out"

class FeaturesTest(unittest.TestCase):
    vehicles = []
    non_vehicles = []
    
    def setUp(self):
        if not os.path.exists(TEST_OUT_DIR):
            os.makedirs(TEST_OUT_DIR)
        self.vehicles = image_util.findImageFilesDeep(LABELED_DATA_VEHICLE_DIR)
        self.non_vehicles = image_util.findImageFilesDeep(LABELED_DATA_NON_VEHICLE_DIR)
            
    def tearDown(self):
        return

    def hogChannels(self, img, orientations, pix_per_cell, cell_per_block, color ):
        img = np.copy(img)
        img = img.astype(np.float32)/255
        if color != None:
            img = image_util.convert_color(img, color)
        ch1 = img[:,:,0]
        ch2 = img[:,:,1]
        ch3 = img[:,:,2]
        features_c1, hog_image_c1 = uut.get_hog_features(ch1, orient=orientations, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,vis=True, feature_vec=True)
        features_c2, hog_image_c2 = uut.get_hog_features(ch2, orient=orientations, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,vis=True, feature_vec=True)
        features_c3, hog_image_c3 = uut.get_hog_features(ch3, orient=orientations, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,vis=True, feature_vec=True)
        return hog_image_c1, hog_image_c2, hog_image_c3


    def visualize_hog(self, index, orientations, pix_per_cell, cell_per_block, color ):
        vehicle_img = image_util.loadImageRGB(self.vehicles[index])
        hog_vehicle_image_c1, hog_vehicle_image_c2, hog_vehicle_image_c3 = self.hogChannels(vehicle_img, orientations=orientations, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, color=color )
        
        img = image_util.arrangeImages([vehicle_img, hog_vehicle_image_c1, hog_vehicle_image_c2, hog_vehicle_image_c3], ["vehicle","vehicle hog c1","vehicle hog c2","vehicle hog c3"])
        
        image_util.saveImage(img, TEST_OUT_DIR + "/vehicle_hog"+str(index)+"_orient"+str(orientations)+"_pix_per_cell"+str(pix_per_cell)+"_cell_per_block"+str(cell_per_block)+"_"+str(color)+".png")

        
        non_vehicle_img = image_util.loadImageRGB(self.non_vehicles[index])
        hog_non_vehicle_image_c1, hog_non_vehicle_image_c2, hog_non_vehicle_image_c3 = self.hogChannels(non_vehicle_img, orientations=orientations, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, color=color)

        img = image_util.arrangeImages([non_vehicle_img, hog_non_vehicle_image_c1, hog_non_vehicle_image_c2, hog_non_vehicle_image_c3], ["non vehicle","non vehicle hog c1","non vehicle hog c2","non vehicle hog c3"])
        
        image_util.saveImage(img, TEST_OUT_DIR + "/non_vehicle_hog"+str(index)+"_orient"+str(orientations)+"_pix_per_cell"+str(pix_per_cell)+"_cell_per_block"+str(cell_per_block)+"_"+str(color)+".png")





    def test_01_labeled_data(self):
        print("number of vehicles: " + str(len(self.vehicles)))
        print("number of non vehicles: " + str(len(self.non_vehicles)))

    def test_02_example_labeled_data(self):
        vehicle_img = image_util.loadImageRGB(self.vehicles[0])
        non_vehicle_img = image_util.loadImageRGB(self.non_vehicles[0])
        img = image_util.arrangeImages([vehicle_img, non_vehicle_img], ["vehicle","non vehicle"])
        image_util.saveImage(img, TEST_OUT_DIR + "/vehicle_non_vehicle.png")

    def test_03_hog(self):
        for color in {'RGB2YCrCb', None, 'RGB2LUV', 'RGB2HLS', 'RGB2HSV', 'RGB2YUV'}:
            for i in range(1,10):
                for orientations in {5,8,9,12,13}:
                    self.visualize_hog(i, orientations=orientations, pix_per_cell=8, cell_per_block=2, color=color)





if __name__ == '__main__':
    unittest.main()
