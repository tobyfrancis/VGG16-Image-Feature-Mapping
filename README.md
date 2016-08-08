# VGGNet-Image-Visualization

## Synopsis

A tool to create maps of image similarities from VGG16 Features, using tSNE for the reduction of features into a 2-Dimensional space representative of how similar the image features are.

## Installation

As the bare minimum, this tool requires the installation of the following packages:
- numpy, h5py
- sci-kit image and sci-kit learn
- Keras (which requires either TensorFlow or Theano)
- Click

You also need to specify the path to the weights for the VGG16 model. This setting is controlled by the option "-v" in the interface, which is by default set to point to "weights/vgg16_weights.h5". [These weights are available here](https://drive.google.com/a/andrew.cmu.edu/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc).

If you want to enable clustering, which displays images only if they are a part of clusters in the feature-space, you must have HDBScan installed.

## Usage

Currently, this tool is available as a python package using Click for the command line interface. The directory of images must be provided as an argument, in this fashion:
```
python image_map.py [OPTIONS] DIRECTORY
```
Options include the usage of GPU, cnmem, clustering, if you are using color images, etc. For example, this turns on GPU usage with CNMeM:
```
python image_map.py --gpu True --cnmem True DIRECTORY
```
To view the available options, use:
```
python image_map.py -h
```
or
```
python image_map.py --help
```

## Motivation

This tool was originally used to explore datasets of microstructure microscopy images. Traditional feature extraction methods (such as SIFT) were considerably less successful in grouping semantically similar images together (in terms of their microstructural constituents), and thus this tool was created as an alternative method.

## Contributors

This tool was made in collaboration with [Brian DeCost](https://github.com/bdecost) of Carnegie Mellon University.