<p align="center" style="border-radius: 10px">
  <img src="assets/logo.jpg" width="50%" alt="logo"/>
</p>

# 🎬 STREAMING AUTOREGRESSIVE VIDEO GENERATION VIA DIAGONAL DISTILLATION

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2509.22622)
[![Code](https://img.shields.io/badge/GitHub-LongLive-blue)](https://github.com/NVlabs/LongLive)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B)
<!-- [![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=CO1QC7BNvig) -->
<!-- [![Demo](https://img.shields.io/badge/Demo-Page-bron)](https://nvlabs.github.io/LongLive) -->

https://github.com/xuanmingliu/Diagonal-distillation/raw/main/assets/fancy.mp4


## 💡 TLDR: When you input, it only takes about 2.6 seconds to generate a 5-second segment!

**STREAMING AUTOREGRESSIVE VIDEO GENERATION VIA DIAGONAL DISTILLATION [[Paper](https://arxiv.org/abs/2509.22622)]** <br />
[Jinxiu Liu](https://brandon-liu-jx.github.io/), [Xuanming Liu], [Kangfu Mei](https://kfmei.com/), [Yandong Wen](https://ydwen.github.io/), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/), [Weiyang Liu](https://wyliu.com/) <br />

Large pretrained diffusion models have significantly enhanced the quality of generated videos, and yet their use in real-time streaming remains limited. Autoregressive models offer a natural framework for sequential frame synthesis but require heavy computation to achieve high fidelity. Diffusion distillation can compress these models into efficient few-step variants, but existing video distillation approaches largely adapt image-specific methods that neglect temporal dependencies. These techniques often excel in image generation but underperform in video synthesis, exhibiting reduced motion coherence, error accumulation over long sequences, and a latency–quality trade-off. We identify two factors that result in these limitations: insufficient utilization of temporal context during step reduction and implicit prediction of subsequent noise levels in next-chunk prediction (exposure bias). To address these issues, we propose Diagonal Distillation, which operates orthogonally to existing approaches and better exploits temporal information across both video chunks and denoising steps. Central to our approach is an asymmetric generation strategy: more steps early, fewer steps later. This design allows later chunks to inherit rich appearance information from thoroughly processed early chunks, while using partially denoised chunks as conditional inputs for subsequent synthesis. By aligning the implicit prediction of subsequent noise levels during chunk generation with the actual inference conditions, our approach mitigates error propagation and reduces oversaturation in long-range sequences. We further incorporate implicit optical flow modeling to preserve motion quality under strict step constraints. Our method generates a 5-second video in **2.61 seconds** (up to **31 FPS**), achieving a **277.3× speedup** over the undistilled model.

## TABLE OF CONTENTS
1. [News](#news)
2. [Highlights](#highlights)
3. [Introduction](#introduction)
4. [Installation](#installation)
5. [Inference](#inference)
6. [Training](#training)
<!-- 7. [How to contribute](#how-to-contribute) -->
7. [Citation](#citation)
8. [License](#license)
9. [Acknowledgement](#acknowledgement)

## News

## Highlights
1. **Ultra-Fast Short Video Generation**: Real-time 31 FPS short video generation is achieved on the H100 GPU with a 277-times acceleration, outperforming SOTA by 1.53 times.
2. **Asymmetric Denoising Strategy**: Asymmetric diagonal denoising is adopted, with more steps (5 steps) in the early block and fewer steps (2 steps) in the later block processing. The rich information in the early block is utilized to significantly reduce the total number of steps.
3. **Motion-Preserving Distillation**: Through the flow distribution matching technology, the temporal dynamics of the teacher-student model are explicitly aligned under strict step limits to maintain motion fidelity.

## Introduction
<p align="center" style="border-radius: 10px">
  <img src="assets/speed_cropped (8).pdf" width="100%" alt="logo"/>
<strong>Our Diagonal Distillation framework achieves comparable quality to the full-step model while significantly reducing latency. The method yields a 1.88× speedup on 5-second short video generation on a single H100 GPU.</strong>
</p>
<p align="center" style="border-radius: 10px">
  <img src="assets/dia_cropped (7).pdf" width="100%" alt="logo"/>
<strong>Diagonal Denoising with Diagonal Forcing and Progressive Step Reduction. We illustrate our method starting with 5 denoising steps for the first chunk and progressively reducing them to 2 steps by Chunk 7 . For chunks with k ≥ 4, we use a fixed two-step denoising process, reusing the Key-Value (KV) cache from the previous chunk’s last noisy frame. This approach maintains temporal coherence while reducing latency, the pseudo-code is provided in the appendix.</strong>
</p>
<p align="center" style="border-radius: 10px">
  <img src="assets/motion_cropped (3).pdf" width="100%" alt="logo"/>
<strong>(a) Without motion loss shows minimal motion amplitude with only slight object movement; (b) With motion loss demonstrates significantly increased motion amplitude throughout the entire frame, validating our method’s effectiveness.</strong>
</p>



## Installation
**Requirements**

We tested this repo on the following setup:
* Nvidia GPU with at least 40 GB memory (A100, and H100 are tested).
* Linux operating system.
* 64 GB RAM.

Other hardware setup could also work but hasn't been tested.

**Environment**

Create a conda environment and install dependencies:
```
git clone https://github.com/xuanmingliu/Diagonal-distillation.git
cd Diagonal-distillation
conda create -n diagonal_distillation python=3.10 -y
conda activate diagonal_distillation
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Inference
**Download checkpoints**

```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download Efficient-Large-Model/LongLive --local-dir longlive_models
```

**Video Generation**
```
bash inference.sh
```

**GUI demo**
```
python demo.py
```

## Training
**Download checkpoints**

Download Wan2.1-T2V-14B as the teacher model.

```
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir wan_models/Wan2.1-T2V-14B
```

**Download text prompts and ODE initialized checkpoint**
```
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
```
**training**
```
bash training.sh
```


<!-- ## How to contribute
- Make sure to have git installed.
- Create your own [fork](https://github.com/NVlabs/LongLive/fork) of the project.
- Clone the repository on your local machine, using git clone and pasting the url of this project.
- Read both the `Requirements` and `Installation and Quick Guide` sections below.
- Commit and push your changes.
- Make a pull request when finished modifying the project. -->


## Citation
Please consider to cite our paper and this framework, if they are helpful in your research.
```bibtex

```

## License


## Acknowledgement
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing): We built this project based on their code base and algorithms. Thank them for their excellent work.
- [Wan](https://github.com/Wan-Video/Wan2.1): We developed based on this fundamental model and are grateful for their outstanding contributions.
