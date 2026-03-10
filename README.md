<p align="center">
<h1 align="center">STREAMING AUTOREGRESSIVE VIDEO GENERATION VIA DIAGONAL DISTILLATION</h1>
</p>
<p align="center">
  <p align="center">
    <a href="https://brandon-liu-jx.github.io/">Jinxiu Liu</a><sup>1</sup>
    ·
    <a href="">Xuanming Liu</a><sup>2</sup>
    ·
    <a href="https://kfmei.com/">Kangfu Mei</a><sup>3</sup>
    ·
    <a href="https://ydwen.github.io/">Yandong Wen</a><sup>2</sup>
    ·
    <a href="https://faculty.ucmerced.edu/mhyang/">Ming-Hsuan Yang</a><sup>4</sup>
    ·
    <a href="https://wyliu.com/">Weiyang Liu</a><sup>5</sup>
    <br/>
    <sub><sup>1</sup>South China University of Technology</sub>
    <sub><sup>2</sup>Westlake University</sub>
    <sub><sup>3</sup>Johns Hopkins University</sub>
    <sub><sup>4</sup>University of California, Merced</sub>
    <sub><sup>5</sup>The Chinese University of Hong Kong</sub>
  </p>
  <h3 align="center"><a href="">Paper</a> | <a href="https://spherelab.ai/diagdistill">Website</a></h3>
</p>

---

We propose ​Diagonal Distillation, a new method for making high-quality video generation much faster. Current methods are either too slow or create videos with poor motion and errors over time.

<p align="center">
  <video src="./assets/new_demo.mp4" controls width="900">
    Your browser does not support the video tag.
  </video>
</p>

---


<!-- https://github.com/user-attachments/assets/7548c2db-fe03-4ba8-8dd3-52d2c6160739 -->


## Requirements
We tested this repo on the following setup:
* Nvidia GPU with at least 24 GB memory (RTX 4090, A100, and H100 are tested).
* Linux operating system.
* 64 GB RAM.

Other hardware setup could also work but hasn't been tested.

## Installation
Create a conda environment and install dependencies:
```
conda create -n dia python=3.10 -y
conda activate dia
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Quick Start
### Download checkpoints
```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download Efficient-Large-Model/LongLive-1.3B --local-dir ./longlive_models
```

Note:
* **Our model works better with long, detailed prompts** since it's trained with such prompts. We will integrate prompt extension into the codebase (similar to [Wan2.1](https://github.com/Wan-Video/Wan2.1/tree/main?tab=readme-ov-file#2-using-prompt-extention)) in the future. For now, it is recommended to use third-party LLMs (such as GPT-4o) to extend your prompt before providing to the model.
* You may want to adjust FPS so it plays smoothly on your device.
* The speed can be improved by enabling `torch.compile`, [TAEHV-VAE](https://github.com/madebyollin/taehv/), or using FP8 Linear layers, although the latter two options may sacrifice quality. It is recommended to use `torch.compile` if possible and enable TAEHV-VAE if further speedup is needed.

### Inference
Example inference script using the chunk-wise autoregressive checkpoint trained with DMD:
```
bash inference.sh
```

## Training

### Diagonal Distillation Training 
```
bash train_init.sh
```
Our training run uses 600 iterations and completes in under 2 hours using 64 H100 GPUs. By implementing gradient accumulation, it should be possible to reproduce the results in less than 16 hours using 8 H100 GPUs.

## Acknowledgements
This codebase is built on top of the open-source implementation of [LongLive](https://github.com/NVlabs/LongLive) by [Tianwei Yin](https://tianweiy.github.io/) and the [Wan2.1](https://github.com/Wan-Video/Wan2.1) repo.

## Citation
If you find this codebase useful for your research, please kindly cite our paper:
```
<!-- @article{huang2025selfforcing,
  title={Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
  author={Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
  journal={arXiv preprint arXiv:2506.08009},
  year={2025}
} -->
```
