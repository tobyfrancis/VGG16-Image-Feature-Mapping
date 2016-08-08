''' Make image maps in a 2-Dimensional embedding space defined by tSNE'''
import numpy as np
from skimage.transform import resize
from skimage.io import imread

def image_montage(X, images, labels=None, mapsize=8192, thumbsize=256):

    halfthumbsize = int(thumbsize/2)
    map_shape = np.array([mapsize,mapsize])
    imagemap = np.ones(map_shape)

    # rescale max distance from origin to 1
    scale = np.max(np.abs(X[:,0:2]))

    for ids, image in enumerate(images):
        try:
            pos = X[ids][:2]

            label_bool = True
            if labels == None:
                label_bool = False
            elif labels[id] <= -1:
                label_bool = False

            if label_bool:

                im = imread(image, as_grey=True)

                # crop arbitrarily to square aspect ratio
                mindim = min(im.shape)
                cropped = im[:mindim,:mindim]

                # make thumbnail
                thumbnail = resize(cropped, (thumbsize,thumbsize), order=2)

                # map position to image coordinates with buffer region
                x, y = np.round(pos/scale * ((mapsize-(thumbsize+2))/2) + (mapsize/2)).astype(int)

                # place thumbnail into image map
                imagemap[x-(halfthumbsize):x+(halfthumbsize),y-(halfthumbsize):y+(halfthumbsize)] = thumbnail
        except IndexError:
            pass

    return imagemap