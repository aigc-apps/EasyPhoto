
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN rm -rf /etc/localtime && ln -s /usr/share/zoneinfo/Asia/Harbin  /etc/localtime
ENV PYTHONUNBUFFERED 1

RUN apt-get update -y && apt-get install -y build-essential apt-utils google-perftools sox \
    ffmpeg libcairo2 libcairo2-dev libcairo2-dev zip wget curl vim git ca-certificates kmod \
    python3-pip python-is-python3 python3.10-venv aria2 && rm -rf /var/lib/apt/lists/*

RUN pip install numpy==1.23.5 Pillow==9.5.0 mpmath>=0.19 networkx sympy -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu117 -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install xformers==0.0.20 mediapipe manimlib svglib fvcore ffmpeg modelscope ultralytics albumentations==0.4.3 -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install voluptuous toml accelerate>=0.20.3 lion-pytorch chardet lxml pathos cryptography openai boto3 -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install aliyun-python-sdk-core aliyun-python-sdk-alimt insightface==0.7.3 onnx==1.14.0 dadaptation PyExecJS -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install pims gradio==3.32.0 setuptools>=42 blendmodes==2022 basicsr==1.4.2 gfpgan==1.3.8 -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install realesrgan==0.3.0 omegaconf==2.2.3 pytorch_lightning==1.9.4 -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install scikit-image timm==0.6.7 piexif==1.1.3 einops psutil==5.9.5 jsonmerge==1.8.0 --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install clean-fid==0.1.35 resize-right==0.0.2 torchdiffeq==0.2.3 kornia==0.6.7 segment_anything supervision -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install lark==1.1.2 inflection==0.5.1 GitPython==3.1.30 safetensors==0.3.1 fairscale numba==0.57.0 moviepy==1.0.2 transforms3d==0.4.1 -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install httpcore==0.15 fastapi==0.94.0 tomesd==0.1.2 numexpr matplotlib pandas av wandb appdirs lpips dataclasses pyqt6 -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install imageio-ffmpeg==0.4.2 rich gdown onnxruntime==1.15.0 ifnude pycocoevalcap clip-anytorch sentencepiece tokenizers==0.13.3 -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install transformers==4.25.1 trampoline==0.1.2 -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install transparent-background ipython seaborn color_matcher trimesh vispy>=0.13.0 rembg>=2.0.50 py-cpuinfo protobuf -i https://mirrors.aliyun.com/pypi/simple/

RUN wget https://pai-aigc-extension.oss-cn-hangzhou.aliyuncs.com/torchsde.zip -O /tmp/torchsde.zip && \
    cd /tmp && unzip torchsde.zip && cd torchsde && python3 setup.py install && rm -rf /tmp/torchsde*

RUN pip install diffusers==0.18.2 segmentation-refinement send2trash~=1.8 dynamicprompts[attentiongrabber,magicprompt]~=0.29.0 gradio_client==0.2.7 -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install opencv-python onnx onnxruntime modelscope


# download more sdwebui requirements
RUN wget https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/requirements_versions.txt
RUN pip install -r requirements_versions.txt -i https://mirrors.aliyun.com/pypi/simple/

# download openai
RUN mkdir -p /root/.cache/
RUN curl -o /root/.cache/huggingface.zip https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/huggingface.zip
RUN unzip /root/.cache/huggingface.zip -d /root/.cache/

RUN pip install controlnet_aux


# torch model to replace tensorflow model,need install mmcv
# https://mmdetection.readthedocs.io/en/v2.9.0/faq.html
# https://github.com/open-mmlab/mmdetection/issues/6765
RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80' pip install mmcv-full==1.7.0 --index https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install mmdet==2.28.2 --index https://pypi.tuna.tsinghua.edu.cn/simple

# oneflow release has bug,we use 2023.10.20 version
RUN pip uninstall tensorflow tensorflow-cpu xformers accelerate -y
RUN pip install --pre oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu117 --index https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install "transformers==4.27.1" "diffusers[torch]==0.19.3" --index https://pypi.tuna.tsinghua.edu.cn/simple
# onediff release has bug,we use 2023.10.20 version
RUN wget -P /root/.cache http://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/onediff.zip
RUN unzip /root/.cache/onediff.zip -d /root/.cache
RUN cd /root/.cache/onediff && python3 -m pip install -e .

WORKDIR /workspace

RUN pip cache purge