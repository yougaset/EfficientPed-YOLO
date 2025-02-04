
Citypersons - v11 2022-12-10 9:01pm
==============================

This dataset was exported via roboflow.com on September 7, 2023 at 9:26 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 6267 images.
Pedestrians are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 1280x640 (Stretch)

The following augmentation was applied to create 2 versions of each source image:
* Random brigthness adjustment of between -30 and 0 percent
* Random Gaussian blur of between 0 and 2.5 pixels
* Salt and pepper noise was applied to 4 percent of pixels


