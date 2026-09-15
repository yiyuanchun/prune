# 192.168.0.44：REDNet + CEGAR MNIST 验证

日期：2026-09-13。目标：在新服务器复现两种扰动半径、各 50 张图片的实验。

## 部署

- SSH：`gpu@192.168.0.44:22`。
- 项目：`/home/gpu/yyc_projects/parnv-v2`。
- 环境：`conda activate alpha-beta-crown`；Python 实际路径为 `/etc/anaconda3/envs/alpha-beta-crown/bin/python`。
- 本地版本基于提交 `2bba1074c343809ef61e1f7f5d6e26c5d3e6a227`，保留首次正式求解前执行 PGD 的改动，以及前一轮支持 50 张图片和多半径的运行脚本。
- 同步并逐文件校验了 810 个源码及配置文件，更新其中 38 个。服务器已有的额外历史文件和旧结果保留，没有删除式镜像。
- 覆盖前的文件备份：`results/gpu44_sync_20260913/server_before.zip`。
- 同步哈希及变更清单：`results/gpu44_sync_20260913/local_source_sha256.json`、`sync_summary.json`。

为支持不同服务器的模型、数据绝对路径，启动器增加可选参数 `--base-manifest`。核心 CEGAR 和 REDNet 算法没有为迁移而修改。

新服务器的现有 ONNX 模型与上次实验哈希一致。迁移了上次已校验的 NNet、MNIST train 图像与标签，存放在 `data/mnist_reference/`，三个文件的 SHA-256 均与上次实验一致。现有相邻的 `data/models/mnist/mnist_fc_relu_64x3.nnet` 与上次使用的文件不同，因此本次清单明确使用迁入的 `data/mnist_reference/mnist_fc_relu_64x3_from_onnx.nnet`。

本机路径配置：`results/gpu44_sync_20260913/experiment_samples_gpu44.json`。根目录的原有 `experiment_samples.json` 保持与本地一致；在新服务器准备实验时使用本机路径配置。

## 参数

| 参数 | 值 |
|---|---|
| 流程 | REDNet + CEGAR |
| 模型 | MNIST FC ReLU，784→64→64→64→10 |
| 样本 | MNIST train 索引 0–49，共 50 张，全部分类正确 |
| 扰动半径 | 0.02、0.03，两组使用同一份样本索引 |
| 超时 | 每次 Marabou 求解 3600 秒，沿用原有定义 |
| PGD | 40 步、5 个起点、seed=0 |
| 批量精化 | 当前伪反例消失后额外恢复最多 3 个 Merge |
| 执行资源 | CPU；两进程分别绑定 8–11、12–15；数值库线程数 1 |
| REDNet 等价性 | 每个输入盒 128 个检查点，容差 `1e-8` |

依赖版本：Python 3.11.8、PyTorch 2.2.1、NumPy 1.26.4、SciPy 1.11.4、ONNX 1.16.0、ONNX Runtime 1.15.1。与旧服务器存在版本差异，已记录在实验的 `audit/environment.json`，不直接将两台服务器的耗时差异归因于算法。

## 输出位置

实验根目录：

```text
/home/gpu/yyc_projects/parnv-v2/results/rednet_50_gpu44_20260913
```

以下路径相对于此目录：

| 内容 | ε=0.02 | ε=0.03 |
|---|---|---|
| 实际配置 | `samples_eps002.json` | `samples_eps003.json` |
| 正式 CSV | `full_eps002/b3_rednet_narv_eps002.csv` | `full_eps003/b3_rednet_narv_eps003.csv` |
| 正式日志 | `full_eps002.log` | `full_eps003.log` |
| 逐样本结果 | `full_eps002/` | `full_eps003/` |

公共样本索引保存在 `sample_indices.json`，性质文件保存在 `properties/`。源码快照与哈希位于 `audit/source/` 和 `audit/source_hashes.json`。启动信息保存在 `full_launch.json`，运行中的逐样本进度保存在各输出目录的 `progress.json`。

## 验证与状态

新服务器上相关测试：**31 passed**。包括初始 PGD、批量精化、预算边界、REDNet 回归和 CIFAR 输入性质测试。测试日志位于 `audit/tests.log`。

两种半径已分别对样本 0 启动小规模验证。REDNet 等价性检查通过，0.02 的 ReLU 数由 192 降为 109，0.03 由 192 降为 134。小规模验证附加直接 Marabou 求解交叉检查；其额外耗时和调用数不计入生产流程统计。

两项小规模验证均已完成，结果与旧服务器一致：

| 半径 | 样本 0 结论 | 正式求解调用 | PGD 调用 | 流程耗时 | 附加直接求解 |
|---|---|---:|---:|---:|---|
| 0.02 | VERIFIED | 1 | 2 | 58.870 秒 | UNSAT |
| 0.03 | UNSAFE | 0 | 2 | 56.794 秒 | SAT |

0.03 的真实反例另经输入盒、原始 NNet 和原始 ONNX 独立复核：原标签 5，反例预测为 3，最大改变量约 0.027，ONNX 的类别优势约 0.759605。复核文件为 `audit/eps003_original_counterexample_check.json`。

正式后台实验已于 **2026-09-13 18:15:26 +0800** 启动：

| 半径 | PID | 实际样本数 | CPU 集合 |
|---|---:|---:|---|
| 0.02 | 100039 | 50 | 8–11 |
| 0.03 | 100040 | 50 | 12–15 |

已确认两个进程存活，并读取实际运行配置，核实流程为 REDNet、样本为 0–49、半径分别为 0.02/0.03、求解超时为 3600 秒。结束本次交互时，两任务均已开始处理样本 0，尚未完成全部 100 次性质验证。按此前约定，确认后台启动后不等待所有样本结束。

## 运行方式

本次使用的准备命令：

```bash
cd /home/gpu/yyc_projects/parnv-v2
conda activate alpha-beta-crown
python parnv-MC/launch_rednet_50.py prepare \
  --run-root results/rednet_50_gpu44_20260913 \
  --source-commit 2bba1074c343809ef61e1f7f5d6e26c5d3e6a227 \
  --base-manifest results/gpu44_sync_20260913/experiment_samples_gpu44.json
```

该目录已由本次实验创建；重做准备时应换用新的实验目录，避免覆盖结果。当前两组的核心运行入口分别是：

```bash
python -u parnv-MC/b3_mnist.py --route rednet \
  --samples results/rednet_50_gpu44_20260913/samples_eps002.json \
  --output results/rednet_50_gpu44_20260913/full_eps002

python -u parnv-MC/b3_mnist.py --route rednet \
  --samples results/rednet_50_gpu44_20260913/samples_eps003.json \
  --output results/rednet_50_gpu44_20260913/full_eps003
```

实际后台启动由 `launch_rednet_50.py full` 设置 CPU 亲和性、线程数、日志和独立进程会话。查看日志：

```bash
tail -f /home/gpu/yyc_projects/parnv-v2/results/rednet_50_gpu44_20260913/full_eps002.log
tail -f /home/gpu/yyc_projects/parnv-v2/results/rednet_50_gpu44_20260913/full_eps003.log
```
