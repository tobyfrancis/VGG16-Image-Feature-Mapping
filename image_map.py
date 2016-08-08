'''
Credit to Brian DeCost of Carnegie Mellon University for Image-Mapping Functionality
github.com/bdecost
'''
import click
import json

from scripts.find_images import *
from scripts.montage import *
from scripts.load_images import *
from scripts.VGG_keras import *

from skimage.io import imread, imsave
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings = CONTEXT_SETTINGS)
@click.argument('directory', nargs=1, type=click.Path())
@click.option('-g', '--gpu', default=False,
              help='Whether or not to use the GPU')
@click.option('-r', '--color', default=False,
              help='True if Images are Colored, False if RGB')
@click.option('-i', '--image_filename', default='imagemap.png',
              help='Filename the Image Mapping Saves to')
@click.option('-c', '--clustering', default=False,
              help='Whether or not to cluster the image features before mapping')
@click.option('-m', '--min_cluster_size', default=10,
              help='Minimum cluster size for HDBSCAN algorithm')
@click.option('-j', '--json_filename', default='images.json',
              help='Filename of image-search results')
@click.option('-n', '--cnmem', default=False,
              help='If you have cnmem installed or not')
@click.option('-v', '--vgg_weights', default='weights/vgg16_weights.h5',
              help='Path to .h5 file containing the weights for VGG16')
def tsne(directory, gpu, color, image_filename, clustering, min_cluster_size, json_filename, cnmem, vgg_weights):

    directory, gpu, color, image_filename, clustering, min_cluster_size, json_filename, cnmem, vgg_weights = \
        check_input_types(directory, gpu, color, image_filename, clustering, min_cluster_size, json_filename, cnmem, vgg_weights)

    if gpu:
        import os
        if cnmem:
            os.environ["THEANO_FLAGS"] = "device=gpu, floatX=float32, lib.cnmem=0.75"
        else:
            os.environ["THEANO_FLAGS"] = "device=gpu, floatX=float32"
    if clustering:
        import hdbscan

    try:
        with open(json_filename,'r') as f:
            image_dataset = json.load(f)
    except FileNotFoundError:
        image_dataset = image_search(directory, json_filename)

    filenames = list(image_dataset.values())
    print('Loading Images...')
    images = load_images(filenames, color, 224)
    print('Done.')

    print('Loading Model...')
    try:
        model = vggmodel('models/'+vgg_weights)

    print('Done.')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    print('Performing forward pass through VGG Net...')
    features = model.predict(images)
    print('Done.')

    dictionary = {}
    for filename, feature in zip(filenames, features):
        dictionary[filename] = list(feature)

    with open('vgg_features.json', 'w') as f:
        json.dump(dictionary, f)

    pca = PCA(n_components=50)
    print('Performing PCA Dimensionality Reduction...')
    PCA_features = pca.fit_transform(features)
    print('Done.')

    print('Performing TSNE Dimensionality Reduction...')
    tsne = TSNE(n_components=2,learning_rate=250, n_iter = 2000, perplexity=20)
    tSNE_features = tsne.fit_transform(PCA_features)
    print('Done.')
    print('Creating Image Map...')
    if clustering:

        #Clusters on PCA 50-Dimension Features
        #User can switch to "features" variable for more accurate clustering
        #Or "tSNE_features" variable for faster clustering in the 2-dimensional mapping

        cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster.fit(PCA_features)
        labels = cluster.labels_

        imagemap = image_montage(tSNE_features, filenames, labels)

    else:
        imagemap = image_montage(tSNE_features, filenames)

    imsave(image_filename, imagemap.tolist())
    print('Done.')

def check_input_types(directory, gpu, color, image_filename, clustering, min_cluster_size, json_filename, cnmem, vgg_weights):

    if not type(directory) is str:
        raise TypeError('Please specify directory of images with a string')

    if not type(image_filename) is str:
        image_filename = str(image_filename)

    if not type(json_filename) is str:
        image_filename = str(json_filename)

    if not (type(vgg_weights) is str and vgg_weights[-3:]=='.h5'):
        raise TypeError('Please specify VGG Weights Filename with a .h5 File in the models/ directory')

    if not type(min_cluster_size) is int:
        if type(min_cluster_size) is float:
            min_cluster_size = int(min_cluster_size)
        elif type(min_cluster_size) is str:
            try:
                min_cluster_size = int(min_cluster_size)
            except ValueError:
                raise TypeError('Please specify HDBSCAN Minimum Cluster Size with an integer')
        else:
            raise TypeError('Please specify HDBSCAN Minimum Cluster Size with an integer')

    if not type(gpu) is bool:
        if type(gpu) is str:
            if gpu == 'True':
                gpu = True
            elif gpu == 'False':
                gpu = False
            else:
                raise TypeError('Please specify if you are using the GPU with a boolean')
        else:
            raise TypeError('Please specify if you are using the GPU with a boolean')

    if not type(clustering) is bool:
        if type(clustering) is str:
            if clustering == 'True':
                clustering = True
            elif clustering == 'False':
                clustering = False
            else:
                raise TypeError('Please specify if you are clustering image features with a boolean')
        else:
            raise TypeError('Please specify if you are clustering image features with a boolean')

    if not type(color) is bool:
        if type(color) is str:
            if color == 'True':
                color = True
            elif color == 'False':
                color = False
            else:
                raise TypeError('Please specify if you are using color images with a boolean')
        else:
            raise TypeError('Please specify if you are using color images with a boolean')

    if not type(cnmem) is bool:
        if type(cnmem) is str:
            if cnmem == 'True':
                cnmem = True
            elif cnmem == 'False':
                cnmem = False
            else:
                raise TypeError('Please specify if you are using cnmem with a boolean')
        else:
            raise TypeError('Please specify if you are using cnmem with a boolean')

    return directory, gpu, color, image_filename, clustering, min_cluster_size, json_filename, cnmem

if __name__ == '__main__':
    tsne()
