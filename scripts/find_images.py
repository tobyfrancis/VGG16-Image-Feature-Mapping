import os
import json
import fnmatch
import itertools

def image_search(toplevel_dir, images_jsonfile):

    all_images = []
    extensions = ['*.tif', '*.png', '*.jpg']
    for dirpath, dirnames, filenames in os.walk(toplevel_dir):
    
        # construct relative file paths
        filepaths = [os.path.join(dirpath, filename) for filename in filenames]
    
        # filter out images with matching extension
        ims = (fnmatch.filter(filepaths, extension) for extension in extensions)
        ims = list(itertools.chain.from_iterable(ims))
        all_images = all_images + ims
    
    image_paths = {key: image for key,image in enumerate(all_images)}

    with open(images_jsonfile, 'w') as f:
        json.dump(image_paths, f)

    return image_paths
