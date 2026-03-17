<p align="center">
  <img src="assets/logo.jpg" alt="Diagonal Distillation logo" width="380"/>
</p>
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
  <h3 align="center"><a href="https://arxiv.org/abs/2603.09488">Paper</a> | <a href="https://spherelab.ai/diagdistill">Website</a></h3>
</p>

---

We propose ​Diagonal Distillation, a new method for making high-quality video generation much faster. Current methods are either too slow or create videos with poor motion and errors over time.

---

https://github.com/user-attachments/assets/97536e89-b784-45ec-980c-e1318cfda185

## ✨ Highlights


1️⃣ **Diagonal Distillation achieves comparable quality to the full-step model while significantly reducing latency. The method yields a 1.88× speedup on 5-second short video generation on a single H100 GPU.** 

<p align="center">
    <img src="assets/speed_cropped (8)_page-0001.jpg" style="border-radius: 15px">
</p>


2️⃣ **Diagonal Denoising with Diagonal Forcing and Progressive Step Reduction. We give an illustration of our method by starting with five denoising steps for the first chunk and gradually reducing them to two steps by Chunk 7. For chunks with k ≥ 4, we use a fixed two-step denoising process, reusing the Key-Value (KV) cache from the final noisy frame of the preceding chunk. This design preserves temporal coherence while minimizing latency, and the corresponding pseudo-code is provided in the appendix.**

<p align="center">
    <img src="assets/dia_cropped (7)_page-0001.jpg" width=800 style="border-radius: 15px">
</p>


3️⃣ **Comparative visualization of temporal training strategies for autoregressive video generation using Causal DiT. Four panels illustrate: (a) Teacher Forcing (green boxes for ground-truth frames), (b) Diffusion Forcing (red boxes for noisy latents), (c) Self Forcing (red boxes for model’s own predictions), and (d) Di- agonal Forcing (Ours) (mixed green/red boxes in diagonal patterns). Each row represents sequential frame generation, with arrows indicating causal dependencies. The diagonal pattern in (d) highlights the core inno- vation—blending clean past frames with recent model-generated ones to align training/inference distributions. This visual comparison underscores how Diagonal Forcing bridges gaps in robustness and coherence seen in baseline methods.** 

<p align="center">
    <img src="assets/new_cropped (1)_page-0001.jpg" style="border-radius: 15px">
</p>


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

For Stage-1 initialization (`init_ckpt` / `generator_ckpt`), you can use either:

```
# Option A: Self-Forcing ODE init
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .

# Option B: Causal-Forcing init ckpt
huggingface-cli download zhuhz22/Causal-Forcing chunkwise/causal_forcing.pt --local-dir checkpoints
```

Then set `configs/exp_stage1_all4_odeinit.yaml` `generator_ckpt` to one of:
* `checkpoints/ode_init.pt`
* `checkpoints/chunkwise/causal_forcing.pt`

Note:
* **Our model works better with long, detailed prompts** since it's trained with such prompts. We will integrate prompt extension into the codebase (similar to [Wan2.1](https://github.com/Wan-Video/Wan2.1/tree/main?tab=readme-ov-file#2-using-prompt-extention)) in the future. For now, it is recommended to use third-party LLMs (such as GPT-4o) to extend your prompt before providing to the model.
* You may want to adjust FPS so it plays smoothly on your device.
* The speed can be improved by enabling `torch.compile`, [TAEHV-VAE](https://github.com/madebyollin/taehv/), or using FP8 Linear layers, although the latter two options may sacrifice quality. It is recommended to use `torch.compile` if possible and enable TAEHV-VAE if further speedup is needed.

## Training

### Diagonal Distillation Training 
```
bash train_two_stage_ode_then_diag.sh
```
Current codebase training is a two-stage pipeline:

* **Stage 1 (`exp_stage1_all4_odeinit`)**: Initialize from either `checkpoints/ode_init.pt` (Self-Forcing ODE init) or `checkpoints/chunkwise/causal_forcing.pt` (Causal-Forcing init), then run base distillation training to obtain a stable stage-1 checkpoint.
* **Stage 2 (`exp_stage2_diag_from_stage1`)**: Resume from the Stage-1 checkpoint (default: `checkpoint_model_001000/model.pt`, i.e., Stage-1 1000-step checkpoint) and continue training with diagonal-denoising settings for better later-chunk temporal quality.

Our training run uses 600 iterations and completes in under 2 hours using 64 H100 GPUs. By implementing gradient accumulation, it should be possible to reproduce the results in less than 16 hours using 8 H100 GPUs.

## Inference
Use a checkpoint produced by training (for example Stage-2 or Stage-1 output) by setting `generator_ckpt` in `configs/diadistill_inference.yaml`, then run:
```
bash inference.sh
```

## Acknowledgements
This codebase is built on top of the open-source implementation of [LongLive](https://github.com/NVlabs/LongLive) by [yukang2017](https://github.com/yukang2017) and the [Wan2.1](https://github.com/Wan-Video/Wan2.1) repo.

## Citation
If you find this codebase useful for your research, please kindly cite our paper:
```
@misc{liu2026streamingautoregressivevideogeneration,
      title={Streaming Autoregressive Video Generation via Diagonal Distillation},
      author={Jinxiu Liu and Xuanming Liu and Kangfu Mei and Yandong Wen and Ming-HsuanYang and Weiyang Liu},
      year={2026},
      eprint={2603.09488},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.09488},
}
```
