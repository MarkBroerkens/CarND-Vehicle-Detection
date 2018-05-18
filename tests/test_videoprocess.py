import matplotlib
matplotlib.use('Agg')
import unittest
import sys
import os
import numpy as np
import cv2

sys.path.append("..")
import car_finder_pipeline
import videoprocess as uut

VIDEO_CLIP="../input_videos/project_video.mp4"
TEST_OUT_DIR="videoprocess"

class TestVideo(unittest.TestCase):
    
    def setUp(self):
        if not os.path.exists(TEST_OUT_DIR):
            os.makedirs(TEST_OUT_DIR)
    
    def _test_ident(self):
        uut.process("../input_videos/project_video.mp4",TEST_OUT_DIR+"/project_video.mp4",cb,subC=(3,6))
    
    def test_pipe_project(self):
        p = car_finder_pipeline.Pipeline(20)
        uut.process("../input_videos/project_video.mp4",TEST_OUT_DIR+"/L_project_video.mp4",p.process)#,subC=(15,20))

    def _test_pipe_challenge(self):
        p = car_finder_pipeline.Pipeline(20)
        uut.process("../input_videos/challenge_video.mp4",TEST_OUT_DIR+"/L_challenge_video.mp4",p.process)#,subC=(0,15))

    def _test_pipe_harder_challenge(self):
        p = car_finder_pipeline.Pipeline(20)
        uut.process("../input_videos/harder_challenge_video.mp4",TEST_OUT_DIR+"/L_harder_challenge_video.mp4",p.process)#,subC=(0,15))

    def _test_pipe_test_video(self):
        p = car_finder_pipeline.Pipeline(20)
        uut.process("../input_videos/test_video.mp4",TEST_OUT_DIR+"/L_test_video.mp4",p.process)#,subC=(0,15))


def cb(img) :
    return img


if __name__ == '__main__':
    unittest.main()
