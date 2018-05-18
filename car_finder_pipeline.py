import numpy as np
from lesson_functions import slide_window, draw_boxes
from search_and_classify import find_cars
import os
import pickle
import hotspots
from scipy.ndimage.measurements import label

TRAIN_PICKLE_FILE=os.path.dirname(os.path.abspath(__file__))+"/train_pickle.p"

class Pipeline:
    def __init__(self, history_length) :
        # get trained classifier and the parameters that were used from pickle
        with open(TRAIN_PICKLE_FILE, "rb") as f:
            dist_pickle = pickle.load(f)
        self.color_space    = dist_pickle["color_space"] # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient         = dist_pickle["orient"]       # HOG orientations
        self.hist_bins      = dist_pickle["hist_bins"]
        self.spatial_size   = dist_pickle["spatial_size"]
        self.pix_per_cell   = dist_pickle["pix_per_cell"] # HOG pixels per cell
        self.cell_per_block = dist_pickle["cell_per_block"] # HOG cells per block
        self.hog_channel    = dist_pickle["hog_channel"] # Can be 0, 1, 2, or "ALL"
        self.spatial_feat   = dist_pickle["spatial_feat"] # Spatial features on or off
        self.hist_feat      = dist_pickle["hist_feat"] # Histogram features on or off
        self.hog_feat       = dist_pickle["hog_feat"] # HOG features on or off
        self.clf            = dist_pickle["clf"]
        self.scaler         = dist_pickle["X_scaler"]
        self.accu           = dist_pickle["accu"]
        self.time           = dist_pickle["time"]
        self.hotspots       = hotspots.Hotspots(history_length)

        print ("Config")
        print ("color_space: " + str(self.color_space ))
        print ("hist_bins: " + str(self.hist_bins))
        print ("spatial_size: " + str(self.spatial_size))
        print ("orient: " + str(self.orient))
        print ("pix_per_cell: " + str(self.pix_per_cell))
        print ("cell_per_block: " + str(self.cell_per_block))
        print ("hog_channel: " + str(self.hog_channel))
        print ("spatial_feat: " + str(self.spatial_feat))
        print ("hist_feat: " + str(self.hist_feat))
        print ("hog_feat: " + str(self.hog_feat))
        print ("accu: " + str(self.accu))
        print ("time: " + str(self.time))

    def process(self,img) :

        boxes = []
        
        # search with box size 64 * 1 = 64
        ystart = 380
        ystop = 480
        scale = 1.0
        boxes = boxes + find_cars(img, color_space=self.color_space, ystart=ystart, ystop=ystop, scale=scale, svc = self.clf, X_scaler=self.scaler, orient=self.orient, pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, spatial_size=self.spatial_size, hist_bins=self.hist_bins)

        # search with box size 64 * 1.5 = 96
        ystart = 400
        ystop = 600
        scale = 1.5
        boxes = boxes + find_cars(img, color_space=self.color_space,ystart=ystart, ystop=ystop, scale=scale, svc = self.clf, X_scaler=self.scaler, orient=self.orient, pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, spatial_size=self.spatial_size, hist_bins=self.hist_bins)


        # search with box size 64 * 2.5 = 160
        ystart = 500
        ystop = 700
        scale = 2.5
        boxes = boxes + find_cars(img, color_space=self.color_space,ystart=ystart, ystop=ystop, scale=scale, svc = self.clf, X_scaler=self.scaler, orient=self.orient, pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, spatial_size=self.spatial_size, hist_bins=self.hist_bins)


        self.hotspots.add_bboxes(boxes)
        draw_img = self.hotspots.draw_labeled_bboxes_with_history(img)
        
        return draw_img
