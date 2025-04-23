# GenForce_Code

The code for paper "Training Tactile Sensors to Learn Force Sensing from Each Other".

# Overview
A framework, GenForce, that enables transferable force sensing across tactile sensors. GenForce unifies tactile signals into shared marker representations, analogous to cortical sensory encoding, allowing force prediction models trained on one sensor to be transferred to others without the need for exhaustive force data collection.  More Details can be found in our paper.

The GenForce model contains two modules:

* **Marker-to-marker translation model** ([m2m](/m2m)). The m2m module is available to transfer the deformation across arbitrary marker representations. The first step is to train the model to bridge the source sensors and the target sensors using the m2m model.This end-to-end model enables direct translation of marker-based images from source images to generated images with the image style of target sensors while preserving the deformation from source sensors .

* **Force prediction model** ([force](/m2m)). After training m2m model, we can transfer all of the marker images with force labels from the old sensor to new sensors, allowing use the transferred marker images and the existing labels to traing force prediciton models to target sensors.

# Getting Started
## Environment
- Install required denpendencied using our conda env file
```
conda env create -f environment.yaml
```
- Activate the conda environment
```
conda activate genforce
```
## Training for maker-to-marker translation

### Step1. Traning for marker encoder
To training the maker-to-marker translation model, we first train a marker encoder for marker feature extraction. The marker encoder is frozon for the image condition.


## Training for force prediction
