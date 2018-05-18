from lesson_functions import extract_features
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import svm
import time
import pickle
import numpy as np
import os

LABELED_DATA_DIR=os.path.dirname(os.path.abspath(__file__))+"/labeled_data/"


def train(vehicle_files,non_vehicle_files,color_space='YCrCb',orient=9, hog_channel='ALL',
          pix_per_cell=8, cell_per_block=2, hist_bins=32, spatial_size=32, spatial_feat=True,
          hist_feat=True, hog_feat=True):
    
    #print ("get vehicle feautures")
    vehicle_features = extract_features(vehicle_files, color_space,
                                    spatial_size=(spatial_size,spatial_size), hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    
    #print ("get non vehicle feautures")
    non_vehicle_features = extract_features(non_vehicle_files, color_space,
                                       spatial_size=(spatial_size,spatial_size), hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
    
    # Create an array stack, NOTE: StandardScaler() expects np.float64
    X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
    
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))
    
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
                                                        #scaled_X, y, train_size=1000, test_size=100, random_state=rand_state)
                                                        scaled_X, y, test_size=0.2, shuffle=True, random_state=rand_state)
                                                        
    
    t=time.time()
    #parameters = {'kernel': ['rbf'], 'C':[0.001, 0.01, 0.1, 1, 10], 'gamma':[0.001, 0.01, 0.1, 1]}
    #svc = svm.SVC(verbose=True, cache_size=2000)
    #clf = GridSearchCV(svc, parameters, verbose=True, n_jobs=6)
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    #print('Best params = ', clf.best_params_)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    accu = round(clf.score(X_test, y_test), 4)
    #print('Test Accuracy of SVC = ', accu)
    
    # Check the prediction time for a single sample
    t=time.time()
    return clf, X_scaler, accu, len(vehicle_features[0])


