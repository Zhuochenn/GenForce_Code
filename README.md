# GenForce_Code

The code for paper "Training Tactile Sensors to Learn Force Sensing from Each Other".

# Overview
A framework, GenForce, that enables transferable force sensing across tactile sensors. GenForce unifies tactile signals into shared marker representations, analogous to cortical sensory encoding, allowing force prediction models trained on one sensor to be transferred to others without the need for exhaustive force data collection.  More Details can be found in our paper.

The GenForce model contains two modules:

* **Marker-to-marker translation model** ([m2m](/m2m)). The m2m module is available to transfer the deformation across arbitrary marker representations. The first step is to train the model to bridge the source sensors and the target sensors using the m2m model.This end-to-end model enables direct translation of marker-based images from source images to generated images with the image style of target sensors while preserving the deformation from source sensors .

* **Force prediction model** ([force](/m2m)). After training m2m model, we can transfer all of the marker images with force labels from the old sensor to new sensors, allowing to use the transferred marker images and the existing labels to train force prediciton models to target sensors.

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
## 1. Training for maker-to-marker translation

For reproducing, Step1 to Step3 are the process we trained our model for the experiments in our paper.

For utilizing the model on real-world sensor, users just need to collect location paired images as the trajectories used in our paper and finetune the model with the checkpoints in Step 3.


### Step1. Training for marker encoder
- To train the maker-to-marker translation model, we first train a marker encoder for marker feature extraction. 
```
sh m2m/vae/marker_encoder.sh
```
### Step2. Pretraining with simulated data
- We freeze marker encoder for the image condition and pretrain the m2m model with simulated data. 
```
sh m2m/m2m/m2m_sim.sh
```
### Step3. Fintuning with real-world data 
#### homogeneous translation
- Finetuning the m2m model with homogeneous data. 
```
sh m2m/m2m/infer/m2m_homo.sh
```
#### material softness effect
- Finetuning the m2m model with material softness effect data. 
```
sh m2m/m2m/infer/m2m_modulus.sh
```
#### homogeneous translation
- Finetuning the m2m model with homogeneous data. 
```
sh m2m/m2m/infer/m2m_heter.sh
```
## 2. Inference for maker-to-marker translation

Upon training m2m model, we can convert all the images with force labels from the source sensors to target sensors.
- homogeneous translation. 
```
sh m2m/m2m/infer/m2m_infer_homo.sh
```
- material softness effect. 
```
sh m2m/m2m/infer/m2m_infer_modulus.sh
```
- heterogeneous translation. 
```
sh m2m/m2m/infer/m2m_infer_heter.sh
```
## 3. Training for force prediciton models
After inference, we get the generated images and force labels from the source sensors. We can use those data to train the force prediction model for each target sensor.
