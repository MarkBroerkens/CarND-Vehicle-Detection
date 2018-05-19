import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import glob
import os
import numpy as np

def loadImagesRGB(path):
    images_list=[]
    for fn in sorted(glob.glob(path+"/*.jpg")):
        images_list.append(loadImageRGB(fn))
    
    for fn in sorted(glob.glob(path+"/*.png")):
        images_list.append(loadImageRGB(fn))
    
    return images_list

def loadImageRGB(path):
    im = cv2.imread(path)
    img = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    return img

# return the paths of all images that are contained below a given path
def findImageFilesDeep(path):
    image_files_list=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if (file.endswith(".jpg") or file.endswith(".png") ):
                image_files_list.append(os.path.join(root, file))
    return image_files_list

def findImageFilesFlat(path):
    image_files_list=[]
    for fn in sorted(glob.glob(path+"/*.jpg")):
        image_files_list.append(fn)

    for fn in sorted(glob.glob(path+"/*.png")):
        image_files_list.append(fn)
    return image_files_list

def saveImage(img,path) :
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imwrite(path,img)

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

def saveBeforeAfterImages(before_img, before_text, after_img, after_text,path):
    mydpi = 80
    width = int((before_img.shape[1] + after_img.shape[1] + 20) / mydpi)
    height = int((before_img.shape[0] + after_img.shape[0] + 30) / mydpi)
    
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(width,height), dpi=mydpi, frameon=False)
    ax1.imshow(before_img)
    ax1.set_title(before_text, fontsize=30)
    ax2.imshow(after_img)
    ax2.set_title(after_text, fontsize=30)
    figure.savefig(path, bbox_inches='tight',
                   transparent=True,
                   pad_inches=0)
    plt.close(figure)

# inspired by https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
def arrangeImages(images, titles, cols = 1, figsize=(6, 4)):
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    fig = plt.figure(figsize=figsize, dpi=80)
    fig.set_tight_layout(True)
    fig.patch.set_alpha(0)
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        plt.imshow(image)
        a.set_title(title, fontsize=20)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


