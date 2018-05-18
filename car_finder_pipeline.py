import numpy as np
from lesson_functions import slide_window, draw_boxes
from search_and_classify import find_cars
import os
import pickle
import hotspots
from scipy.ndimage.measurements import label

TRAIN_PICKLE_FILE=os.path.dirname(os.path.abspath(__file__))+"/train_pickle.p"

class Pipeline:
    def __init__(self) :
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
        self.hotspots       = hotspots.Hotspots(20)

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


        # draw all windows into single file
#        windows = slide_window(img, x_start_stop=[None, None], y_start_stop=self.y_start_stop,
#                     xy_window=(64, 64), xy_overlap=(0.5, 0.5))
#
#        hot_windows = search_windows(img, windows, self.clf, self.scaler, color_space=self.color_space,
#                                      spatial_size=self.spatial_size, hist_bins=self.hist_bins,
#                                      orient=self.orient, pix_per_cell=self.pix_per_cell,
#                                      cell_per_block=self.cell_per_block,
#                                      hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
#                                      hist_feat=self.hist_feat, hog_feat=self.hog_feat)
#        img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)

        boxes = []
        
        # search with box size 64 * 1 = 64
        ystart = 400
        ystop = 464
        scale = 1.0
        boxes = boxes + find_cars(img, color_space=self.color_space, ystart=ystart, ystop=ystop, scale=scale, svc = self.clf, X_scaler=self.scaler, orient=self.orient, pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, spatial_size=self.spatial_size, hist_bins=self.hist_bins)

        ystart = 416
        ystop = 480
        scale = 1.0
        boxes = boxes + find_cars(img, color_space=self.color_space,ystart=ystart, ystop=ystop, scale=scale, svc = self.clf, X_scaler=self.scaler, orient=self.orient, pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, spatial_size=self.spatial_size, hist_bins=self.hist_bins)

        # search with box size 64 * 1.5 = 96
        ystart = 400
        ystop = 496
        scale = 1.5
        boxes = boxes + find_cars(img, color_space=self.color_space,ystart=ystart, ystop=ystop, scale=scale, svc = self.clf, X_scaler=self.scaler, orient=self.orient, pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, spatial_size=self.spatial_size, hist_bins=self.hist_bins)

        ystart = 432
        ystop = 528
        scale = 1.5
        boxes = boxes + find_cars(img, color_space=self.color_space,ystart=ystart, ystop=ystop, scale=scale, svc = self.clf, X_scaler=self.scaler, orient=self.orient, pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, spatial_size=self.spatial_size, hist_bins=self.hist_bins)

        # search with box size 64 * 2 = 128
        ystart = 400
        ystop = 528
        scale = 2.0
        boxes = boxes + find_cars(img, color_space=self.color_space,ystart=ystart, ystop=ystop, scale=scale, svc = self.clf, X_scaler=self.scaler, orient=self.orient, pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, spatial_size=self.spatial_size, hist_bins=self.hist_bins)

        ystart = 432
        ystop = 560
        scale = 2.0
        boxes = boxes + find_cars(img, color_space=self.color_space,ystart=ystart, ystop=ystop, scale=scale, svc = self.clf, X_scaler=self.scaler, orient=self.orient, pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, spatial_size=self.spatial_size, hist_bins=self.hist_bins)

        # search with box size 64 * 3.5 = 196
        ystart = 400
        ystop = 596
        scale = 3.5
        boxes = boxes + find_cars(img, color_space=self.color_space,ystart=ystart, ystop=ystop, scale=scale, svc = self.clf, X_scaler=self.scaler, orient=self.orient, pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, spatial_size=self.spatial_size, hist_bins=self.hist_bins)

        ystart = 464
        ystop = 660
        scale = 3.5
        boxes = boxes + find_cars(img, color_space=self.color_space,ystart=ystart, ystop=ystop, scale=scale, svc = self.clf, X_scaler=self.scaler, orient=self.orient, pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, spatial_size=self.spatial_size, hist_bins=self.hist_bins)


        self.hotspots.add_bboxes(boxes)
        # Visualize the heatmap when displaying
        #heatmap = np.clip(heat, 0, 255)
        
        draw_img = self.hotspots.draw_labeled_bboxes_with_history(img)
        
        #img = draw_boxes(img, boxes1, color=(0, 0, 255), thick=6)
        #img = draw_boxes(img, boxes2, color=(0, 255, 0), thick=6)
        #img = draw_boxes(img, boxes3, color=(255, 0, 0), thick=6)
        
        return draw_img
