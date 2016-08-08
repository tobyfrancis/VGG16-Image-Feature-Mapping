'''Loads Images from JSON file, crops them, and applies Normalization for VGGNet'''
import numpy as np
from skimage.io import imread, imsave
from skimage.color import gray2rgb

def load_images(filenames, rgb, crop=None):

    images = []
    for filename in filenames:
        try:

            if rgb:
                image = imread(filename, as_grey=False)
            else:
                image = imread(filename, as_grey = True)
            x2, y2 = image.shape
            if not crop:
                crop = min(x2,y2) - min(x2,y2)%8
            if x2 > crop and y2 > crop:
                x1 = (x2-crop)/2
                x2 = x2 - x1
                y1 = 0
                y2 = copy.deepcopy(crop)
                image = image[x1:x2,y1:y2]
                image = np.array(image).astype(float)
                if not rgb:
                    image = gray2rgb(image)

                #Applying the normalization suggested by the authors of VGGNet
                image[:,:,0] -= 103.939
                image[:,:,1] -= 116.779
                image[:,:,2] -= 123.68
                images.append(image.T[0:3])

        except FileNotFoundError:
            pass

    return np.array(images)