# LongLive 两阶段继续训练手册（基于 NVlabs/LongLive 已训练模型）

本文档对应当前仓库代码状态（你当前最终版本），目标是：

1. **Stage-1**：先把 MSE regression target 学稳，且 target 来自 teacher 4-step clean 去噪结果。
2. **Stage-2**：在 Stage-1 权重基础上再训（默认 1000 steps，500 step 存一次），提升 later chunks 质量。

---

## 1. 当前代码关键点（已对齐你的要求）

当前代码已经包含以下关键修正：

1. **Regression target 是 clean 的**
   - `pipeline/self_forcing_training.py` 里先把 `denoised_pred` 写入输出，再单独 `add_noise` 做 cache refresh。
   - 即：regression target 不会拿到 add_noise 后的缓存张量。

2. **Teacher 4-step regression 用最后一步 clean 输出**
   - `model/dmd.py` 的 teacher rollout 路径会临时强制 `last_step_only=True`，避免随机 early-exit 导致 target 变噪。

3. **训练侧 denoising schedule 可与推理侧对齐**
   - `pipeline/self_forcing_training.py` 已接入 block-level schedule（4/3/2/2...）。
   - 由 `use_diagonal_denoising` 控制是否启用该分块 schedule。

---

## 2. 环境与入口

### 2.1 环境

训练使用你当前稳定环境：

- conda env: `dia`
- 典型启动方式：

```bash
cd /path/to/dia
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dia
```

### 2.2 脚本入口

- Stage-1 常用脚本：`train_init.sh`（默认读 `configs/diadistill_train_init.yaml`）
- Stage-2 常用脚本：建议继续用 `train_init.sh`，但切到 Stage-2 配置文件（下文给出方式）

---

## 3. 基于 NVlabs/LongLive 已训练模型继续训练

核心就是把配置里的 `generator_ckpt` 指到你已有的基础模型，例如：

```yaml
generator_ckpt: /path/to/your/base_or_previous_stage_checkpoint/model.pt
```

你当前目录中的 run 输出一般在：

- `outputs/<run_id>/checkpoint_model_xxxxxx/model.pt`

建议每个阶段使用**新的输出目录**（脚本已自动按时间戳新建）。

---

## 4. 两阶段训练建议

## 4.1 Stage-1（先稳住 regression target）

目标：让 regression 分支先学到稳定 clean target（teacher 4-step）。

建议关键设置（在 `configs/diadistill_train_init.yaml` 或复制出的 stage1 配置里）：

```yaml
use_teacher_4step_regression: true
teacher_4step_list: [1000, 750, 500, 250]
reg_loss_type: mse

# 可先偏重 regression（示例）
lambda_reg: 1.0
lambda_spatial_dmd: 4.0
lambda_flow_dmd: 0.0
gamma_temporal: 0.0
use_flow_reg_loss: false
use_motion_loss: false
```

说明：
- 这组更偏向“先把 clean target 对齐”。
- 你也可以保留少量 flow 项，但如果出现 target 不稳，优先先关 flow 分支。

启动：

```bash
cd /path/to/dia
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dia
bash train_init.sh
```

---

## 4.2 Stage-2（你现在这阶段：提升 later chunk 质量）

目标：在 Stage-1 权重基础上强化时序/后段 chunk 质量。

你当前“最终版”思路可用下面配置：

```yaml
# 从 Stage-1 best ckpt 接着训
generator_ckpt: /path/to/stage1_best/model.pt

# 训练长度与保存
max_iters: 1000
log_iters: 500      # 当前代码保存触发与 log_iters 同步
max_checkpoints: 5

# 你当前常用 DMD 权重组合
lambda_spatial_dmd: 4.0
lambda_flow_dmd: 4.0
gamma_temporal: 1.0
use_flow_reg_loss: true
use_motion_loss: true

# regression teacher 分支（建议继续开）
use_teacher_4step_regression: true
teacher_4step_list: [1000, 750, 500, 250]

# 上下文噪声（你当前常用）
context_noise: 100
```

如果你希望严格“就是当前配置+改训练长度/保存频率”，只改这两项即可：

```yaml
max_iters: 1000
log_iters: 500
```

启动同 Stage-1：

```bash
bash train_init.sh
```

---

## 5. 推荐的配置管理方式（避免覆盖）

建议保留两份显式配置：

- `configs/longlive_train_stage1.yaml`
- `configs/longlive_train_stage2.yaml`

然后改脚本里的 `CONFIG=...` 或临时命令行指定：

```bash
torchrun --nproc_per_node=8 --master_port 29xxx \
  train.py --config_path configs/longlive_train_stage2.yaml \
  --logdir outputs/<run_id> \
  --wandb-save-dir outputs/<run_id> \
  --disable-wandb
```

---

## 6. 关键可调参数说明（按影响分组）

## 6.1 Checkpoint / 继续训练

- `generator_ckpt`
  - 训练起点权重路径。
  - Stage-2 必须指向 Stage-1 产出的 checkpoint。

- `max_iters`
  - 总训练步数。

- `log_iters`
  - 日志与 checkpoint 的触发周期（你当前代码里保存与它同步）。

- `max_checkpoints`
  - 最多保留 ckpt 数量，超出会滚动删除旧的。

## 6.2 Regression 相关（你的核心）

- `use_teacher_4step_regression`
  - 是否使用 teacher 4-step rollout 作为 regression target 来源。

- `teacher_4step_list`
  - teacher rollout 的固定步列表，常用 `[1000, 750, 500, 250]`。

- `lambda_reg`
  - regression loss 权重。
  - Stage-1 可适当调大；Stage-2 可按效果降到 0~1。

- `reg_loss_type`
  - `mse | charbonnier | cauchy`。
  - 你当前目标是 MSE 稳定，优先 `mse`。

## 6.3 Flow / 时序相关

- `use_flow_reg_loss`
  - 是否启用 flow regression 分支（motion head feature 对齐）。

- `use_motion_loss`
  - 是否启用 DMD motion loss。

- `lambda_flow_dmd`
  - flow DMD 项权重。

- `gamma_temporal`
  - temporal 总系数（包含 flow 项整体缩放）。

- `flow_reg_ema_decay`
  - motion head teacher EMA 更新衰减。

## 6.4 Chunk / Context / Denoising schedule

- `context_noise`
  - cache refresh 的噪声等级。
  - 你当前常用 `100`。

- `denoising_step_list`
  - 主 denoising 步列表（例如 `[1000, 100]`）。

- `use_diagonal_denoising`
  - 打开后，训练按 block schedule（4/3/2/2...）执行。

- `warmup_mid_steps_raw`
  - diagonal schedule 的中间步候选（只有启用 diagonal 时有意义）。

- `num_frame_per_block`
  - 每个 chunk 的帧数（当前常用 3）。

- `slice_last_frames`
  - 末尾参与训练/反传的帧段设置。

## 6.5 优化器与稳定性

- `lr`, `lr_critic`
  - 生成器/critic 学习率。

- `batch_size`, `gradient_accumulation_steps`, `total_batch_size`
  - 等效 batch 相关；显存不够优先调这组。

- `mixed_precision`, `gradient_checkpointing`
  - 显存与速度权衡。

- `model_kwargs.local_attn_size`, `model_kwargs.sink_size`
  - 注意力窗口与上下文策略，影响长视频质量和显存。

---

## 7. 产物与日志定位

每次训练都会在输出目录生成：

- `train.log`
- `train_config_input.yaml`
- `launch_info.txt`
- `checkpoint_model_xxxxxx/model.pt`

默认输出根目录：

- `outputs`

快速看训练状态：

```bash
tail -n 200 outputs/<run_id>/train.log
```

---

## 8. 常见排查

1. 看起来没保存 ckpt
- 检查 `log_iters` 是否过大。
- 检查是否真的跑过对应 step（`train.log`）。

2. regression 不稳定/像噪声
- 确认 `use_teacher_4step_regression: true`。
- 优先提高 `lambda_reg`，并暂时减弱/关闭 flow 项（`gamma_temporal`, `lambda_flow_dmd`, `use_flow_reg_loss`）。

3. later chunk 质量差
- Stage-2 中逐步提高 `gamma_temporal` / `lambda_flow_dmd`。
- 结合 `context_noise`（常见 0/50/100）做网格实验。

---

## 9. 建议的最小实验矩阵

若你要快速找最优组合，建议优先扫这 3 维：

1. `lambda_reg`: `0.0 / 0.5 / 1.0`
2. `gamma_temporal`: `0.0 / 0.5 / 1.0`
3. `context_noise`: `0 / 50 / 100`

其余参数固定，先看 10 prompts 的主观质量与后段 chunk 稳定性，再决定是否扩展搜索。

---

如果后续你决定固定 Stage-1/Stage-2 的最终参数，我可以再帮你把两份配置文件 (`longlive_train_stage1.yaml`, `longlive_train_stage2.yaml`) 和对应启动脚本一次性落好。
