from moviepy.editor import VideoFileClip

def process(clipFile, outputFile,cb,subC=None) :
    clip1 = VideoFileClip(clipFile)
    if(subC != None) :
        clip1 = clip1.subclip(subC[0],subC[1])

    trafoClip = clip1.fl_image(cb) #NOTE: this function expects color images!!
    trafoClip.write_videofile(outputFile, audio=False)

