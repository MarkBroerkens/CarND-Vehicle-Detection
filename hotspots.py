import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label


class Hotspots:
    def __init__(self, history_max_size):
        # history of rectangles previous n frames
        self.history = []
        self.history_max_size = history_max_size
    
    def add_bboxes(self, bbox_list):
        self.history.append(bbox_list)
        if len(self.history) > self.history_max_size:
            # throw out oldest set of bboxes
            self.history = self.history[len(self.history)-self.history_max_size:]
    
    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        
        # Return updated heatmap
        return heatmap

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap


    def draw_labeled_bboxes_with_history(self, img):
        heatmap_img = np.zeros_like(img[:,:,0])
        
        for bbox_list in self.history:
            heatmap = self.add_heat(heatmap_img, bbox_list)
        heatmap = self.apply_threshold(heatmap, self.history_max_size * 1.5)
        labels = label(heatmap)
        draw_img = self.draw_labeled_bboxes(img, labels)
        #draw_img = self.draw_heatmap(heatmap)
        return  draw_img

    def draw_heatmap(self,heatmap):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        plt.imshow(heatmap, cmap='hot')
        fig.canvas.draw()
        fig.tight_layout()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return  img

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img
