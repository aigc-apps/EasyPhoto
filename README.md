# 📷 EasyPhoto | Your Smart AI Photo Generator.
🦜 EasyPhoto is a tool for generating AI portraits that can be used to train digital doppelgangers relevant to you. 

🦜 🦜 Welcome!

English | [简体中文](./README_zh-CN.md)

# Table of Contents
- [Introduction](#introduction)
- [TODO List](#todo-list)
- [Quick Start](#quick-start)
    - [1. Local install: Check/Downloading/Installation](#1-local-install-environment-checkdownloadinginstallation)
- [How to use](#how-to-use)
    - [1. Model Training](#1-model-training)
    - [2. Inference](#2-inference)
- [Algorithm Detailed](#algorithm-detailed)
    - [1. Architectural Overview](#1-architectural-overview)
    - [2. Training Detailed](#2-training-detailed)
    - [3. Inference Detailed](#3-inference-detailed)
- [Reference](#reference)
- [Related Project](#Related-Project)
- [License](#license)
- [ContactUS](#contactus)

# Introduction
EasyPhoto is a tool for generating AI portraits that can be used to train digital doppelgangers relevant to you. Training is recommended to be done with 5 to 20 portrait images, preferably half-body photos and do not wear glasses (It doesn't matter if the characters in a few pictures wear glasses). After the training is done, we can generate it in the Inference section. We support using preset template images or uploading your own images for Inference.    

Please read our Contributor Covenant [covenant](./COVENANT.md) | [简体中文](./COVENANT_zh-CN.md).   

What's New: 
- We also support the [Stable Diffusion Webui Plug-In](https://github.com/aigc-apps/sd-webui-EasyPhoto/). [🔥 2023.09.19]
- Create Code! Support for Windows and Linux Now. Support multi-people generation. Support different base models for training and inference. Support fine-tuning the background and calculating similarity. [🔥 2023.09.19]

These are our generated results:
![results_1](images/results_1.jpg)
![results_2](images/results_2.jpg)
![results_3](images/results_3.jpg)

Our ui interface is as follows:  
**train part:**
![train_ui](images/train_ui.jpg)
**inference part:**
![infer_ui](images/infer_ui.jpg)

# TODO List
- Support chinese ui.
- Support change in template's background.
- Support high resolution.

# Quick Start
### 1. Local install: Environment Check/Downloading/Installation
#### a. Environment Check
We have verified EasyPhoto execution on the following environment:  

The detailed of Windows 10:  
- OS: Windows10
- python: py3.10
- pytorch: torch2.0.1
- tensorflow-cpu: 2.13.0
- CUDA: 11.7
- CUDNN: 8+
- GPU: Nvidia-3060 12G

The detailed of Linux:  
- OS: Ubuntu 20.04, CentOS
- python: py3.10 & py3.11
- pytorch: torch2.0.1
- tensorflow-cpu: 2.13.0
- CUDA: 11.7
- CUDNN: 8+
- GPU: Nvidia-A10 24G & Nvidia-V100 16G & Nvidia-A100 40G

We need about 60GB available on disk (for saving weights and datasets process), please check!

#### b. Download and Installation
```
# Download and Installation
git clone https://github.com/aigc-apps/EasyPhoto.git
cd EasyPhoto
pip install -r requirements.txt

# launch tool
python app.py
```

### 2. Build from Docker

```
# pull image
docker pull registry.cn-shanghai.aliyuncs.com/pai-ai-test/eas-service:easyphoto-diffusers-py310-torch210-cu117

# enter image
docker run -it -p 7860:7860 --network host --gpus all registry.cn-shanghai.aliyuncs.com/pai-ai-test/eas-service:easyphoto-diffusers-py310-torch210-cu117

# launch
python app.py
```

# How to use
### 1. Model Training
The EasyPhoto training interface is as follows:

- On the left is the training image. Simply click Upload Photos to upload the image, and click Clear Photos to delete the uploaded image;
- On the right are the training parameters, which cannot be adjusted for the first training.

After clicking Upload Photos, we can start uploading images. **It is best to upload 5 to 20 images here, including different angles and lighting conditions**. It is best to have some images that do not include glasses. If they are all glasses, the generated results may easily generate glasses.
![train_1](images/train_1.jpg)

Then we click on "Start Training" below, and at this point, we need to fill in the User ID above, such as the user's name, to start training.
![train_2](images/train_2.jpg)

After the model starts training, the webui will automatically refresh the training log. If there is no refresh, click Refresh Log button.
![train_3](images/train_3.jpg)

If you want to set parameters, the parsing of each parameter is as follows:

|Parameter Name | Meaning|
|--|--|
|Resolution | The size of the image fed into the network during training, with a default value of 512|
|Validation & save steps | The number of steps between validating the image and saving intermediate weights, with a default value of 100, representing verifying the image every 100 steps and saving the weights|
|Max train steps | Maximum number of training steps, default value is 800|
|Max steps per photos | The maximum number of training sessions per image, default to 200|
|Train batch size | The batch size of the training, with a default value of 1|
|Gradient accumulation steps | Whether to perform gradient accumulation. The default value is 4. Combined with the train batch size, each step is equivalent to feeding four images|
|Dataloader num workers | The number of jobs loaded with data, which does not take effect under Windows because an error will be reported if set, but is set normally on Linux|
|Learning rate | Train Lora's learning rate, default to 1e-4|
|Rank Lora | The feature length of the weight, default to 128|
|Network alpha | The regularization parameter for Lora training, usually half of the rank, defaults to 64|

### 2. Inference 
#### a. single people
- Step 1: Click the refresh button to query the model corresponding to the trained user ID.
- Step 2: Select the user ID.
- Step 3: Select the template that needs to be generated.
- Step 4: Click the Generate button to generate the results.

![single_people](images/single_people.jpg)

#### b. multi people
- Step 1: Go to the settings page of EasyPhoto and set num_of_faceid is greater than 1.
- Step 2: Apply settings.
- Step 3: Restart the ui interface of the webui.
- Step 4: Return to EasyPhoto and upload the two person template.
- Step 5: Select the user IDs of two people.
- Step 6: Click the Generate button. Perform image generation.

![multi_people](images/multi_people.jpg)

# Algorithm Detailed

### 1. Architectural Overview

![overview](images/overview.jpg)

In the field of AI portraits, we expect model-generated images to be realistic and resemble the user, and traditional approaches introduce unrealistic lighting (such as face fusion or roop). To address this unrealism, we introduce the image-to-image capability of the stable diffusion model. Generating a perfect personal portrait takes into account the desired generation scenario and the user's digital doppelgänger. We use a pre-prepared template as the desired generation scene and an online trained face LoRA model as the user's digital doppelganger, which is a popular stable diffusion fine-tuning model. We use a small number of user images to train a stable digital doppelgänger of the user, and generate a personal portrait image based on the face LoRA model and the expected generative scene during inference. 

### 2. Training Detailed

![overview](images/train_detail.jpg)

First, we perform face detection on the input user image, and after determining the face location, we intercept the input image according to a certain ratio. Then, we use the saliency detection model and the skin beautification model to obtain a clean face training image, which basically consists of only faces. Then, we label each image with a fixed label. There is no need to use a labeler here, and the results are good. Finally, we fine-tune the stabilizing diffusion model to get the user's digital doppelganger.   

During training, we utilize the template image for verification in real time, and at the end of training, we calculate the face id gap between the verification image and the user's image to achieve Lora fusion, which ensures that our Lora is a perfect digital doppelganger of the user.

In addition, we will choose the image that is most similar to the user in the validation as the face_id image, which will be used in Inference.

### 3. Inference Detailed
#### a. First Diffusion:  
First, we will perform face detection on our incoming template image to determine the mask that needs to be inpainted for stable diffusion. then we will use the template image to perform face fusion with the optimal user image. After the face fusion is completed, we use the above mask to inpaint (fusion_image) with the face fused image. In addition, we will affix the optimal face_id image obtained during training to the template image by affine transformation (replaced_image). Then we will apply Controlnets on it, we use canny with color to extract features for fusion_image and openpose for replaced_image to ensure the similarity and stability of the images. Then we will use Stable Diffusion combined with the user's digital split for generation.

#### b. Second Diffusion:
After getting the result of First Diffusion, we will fuse the result with the optimal user image for face fusion, and then we will use Stable Diffusion again with the user's digital doppelganger for generation. The second generation will use higher resolution.

# Reference
- insightface：https://github.com/deepinsight/insightface    
- cv_resnet50_face：https://www.modelscope.cn/models/damo/cv_resnet50_face-detection_retinaface/summary  
- cv_u2net_salient：https://www.modelscope.cn/models/damo/cv_u2net_salient-detection/summary 
- cv_unet_skin_retouching_torch：https://www.modelscope.cn/models/damo/cv_unet_skin_retouching_torch/summary   
- cv_unet-image-face-fusion：https://www.modelscope.cn/models/damo/cv_unet-image-face-fusion_damo/summary  
- kohya：https://github.com/bmaltais/kohya_ss
- controlnet-webui：https://github.com/Mikubill/sd-webui-controlnet

# Related Project
We've also listed some great open source projects as well as any extensions you might be interested in:
- [ModelScope](https://github.com/modelscope/modelscope).
- [FaceChain](https://github.com/modelscope/facechain).
- [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet).
- [sd-webui-roop](https://github.com/s0md3v/sd-webui-roop).
- [roop](https://github.com/s0md3v/roop).
- [sd-webui-deforum](https://github.com/deforum-art/sd-webui-deforum).
- [sd-webui-additional-networks](https://github.com/kohya-ss/sd-webui-additional-networks).
- [a1111-sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete).
- [sd-webui-segment-anything](https://github.com/continue-revolution/sd-webui-segment-anything).
- [sd-webui-tunnels](https://github.com/Bing-su/sd-webui-tunnels).
- [sd-webui-mov2mov](https://github.com/Scholar01/sd-webui-mov2mov).

# License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).

# ContactUS
1. Use [Dingding](https://www.dingtalk.com/) to search group 38250008552 or Scan to join
2. Since the WeChat group is full, you need to scan the image on the right to add this student as a friend first, and then join the WeChat group.

<figure>
<img src="images/erweima.jpg" width=300/>
<img src="images/wechat.jpg" width=300/>
</figure>