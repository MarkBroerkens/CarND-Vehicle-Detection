import sys
import os
import unittest
import cv2
import pickle
import time

sys.path.append("..")
import image_util
import train as uut


LABELED_DATA_VEHICLE_DIR="../labeled_data/vehicles"
LABELED_DATA_NON_VEHICLE_DIR="../labeled_data/non-vehicles"

class ImageUtilTest(unittest.TestCase):
    def setUp(self):
        self.vehicles = image_util.findImageFilesDeep(LABELED_DATA_VEHICLE_DIR)
        self.non_vehicles = image_util.findImageFilesDeep(LABELED_DATA_NON_VEHICLE_DIR)

        return
    
    def tearDown(self):
        return

    def _test_01_train_histogram(self):
 
        print ("histogram")
        print ("len features | color_space | nbins | accuracy")
        for color_space in ['RGB','HSV','LUV','HLS','YUV','YCrCb']:
            for nbins in [8,16,32,64]:
                clf,X_scaler,accu,len = uut.train(self.vehicles, self.non_vehicles,color_space=color_space, hist_bins=nbins, spatial_feat=False, hist_feat=True, hog_feat=False)
                print("%s | %s | %d | %0.3f " % (len, color_space, nbins,accu))

    def _test_02_train_spatial(self):
        
        print ("spatial")
        print ("len features | color_space | spatial_size | accuracy")
        for color_space in ['RGB','HSV','LUV','HLS','YUV','YCrCb']:
            for spatial_size in [8,16,32,64]:
                clf,X_scaler,accu,len = uut.train(self.vehicles, self.non_vehicles,color_space=color_space, spatial_size=spatial_size, spatial_feat=True, hist_feat=False, hog_feat=False)
                print("%s | %s | %d | %0.3f " % (len, color_space, spatial_size,accu))

    def _test_03_train_hog(self):
        
        print ("hog")
        print ("| len features | color_space | orient | pix_per_cell | cell_per_block | hog_channel | accuracy | " )
        for color_space in ['HSV','LUV','HLS','YUV','YCrCb']:
            for orient in [5,9,13]:
                for pix_per_cell in [8,16]:
                    for cell_per_block in [2,4]:
                        for hog_channel in ['ALL']:
                            clf,X_scaler,accu,len = uut.train(self.vehicles, self.non_vehicles,color_space=color_space, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=False, hist_feat=False, hog_feat=True)
                            print("| %d | %s | %d | %d | %d | %s | %0.3f |" % (len, color_space, orient, pix_per_cell, cell_per_block, hog_channel ,accu))

    def _test_04_train_complete(self):
        
        print ("hog")
        print ("| len features | spatial_feat | hist_feat | hog_feat | accuracy |" )
        for spatial_feat in [True, False]:
            for hist_feat in [True, False]:
                for hog_feat in [True, False]:
                    clf,X_scaler,accu,len = uut.train(self.vehicles, self.non_vehicles,color_space='YCrCb', hist_bins=16, spatial_size=16, orient=5, pix_per_cell=16, cell_per_block=4, hog_channel='ALL', spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
                    print("| %d | %s | %s | %s | %0.3f |" % (len, spatial_feat, hist_feat, hog_feat,accu))

    def test_05_train_and_save(self):
        color_space='YUV'
        hist_bins=64
        spatial_size=16
        orient=13
        pix_per_cell=16
        cell_per_block=2
        hog_channel='ALL'
        spatial_feat=True
        hist_feat=True
        hog_feat=True
        
        clf,X_scaler,accu,len = uut.train(self.vehicles, self.non_vehicles,color_space=color_space, hist_bins=hist_bins, spatial_size=spatial_size, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        print("| %d | %s | %s | %s | %0.3f |" % (len, spatial_feat, hist_feat, hog_feat,accu))
        # Save the training result for later use
        dist_pickle = {}
        dist_pickle["color_space"] = color_space
        dist_pickle["hist_bins"] = hist_bins
        dist_pickle["spatial_size"] = spatial_size
        dist_pickle["orient"] = orient
        dist_pickle["pix_per_cell"] = pix_per_cell
        dist_pickle["cell_per_block"] = cell_per_block
        dist_pickle["hog_channel"] = hog_channel
        dist_pickle["spatial_feat"] = spatial_feat
        dist_pickle["hist_feat"] = hist_feat
        dist_pickle["hog_feat"] = hog_feat
        dist_pickle["clf"] = clf
        dist_pickle["X_scaler"] = X_scaler
        dist_pickle["accu"] = accu
        dist_pickle["feature_len"] = len
        dist_pickle["time"] = time.time()
        with open("train_pickle.p", "wb") as f:
            pickle.dump( dist_pickle, f )

if __name__ == '__main__':
    unittest.main()
