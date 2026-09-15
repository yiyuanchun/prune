# REDNet + CEGAR：50 个 MNIST 样本、两种扰动半径

日期：2026-09-13。

## 代码与同步

本地提交 `2bba1074c343809ef61e1f7f5d6e26c5d3e6a227`（Run PGD before first Marabou call in CEGAR）中的修改已保留：在第一次正式求解前执行 PGD。

为执行本次实验，对 `parnv-MC/b3_mnist.py` 进行了必要适配：从样本清单读取数量、从扰动半径生成 CSV 文件名、在 CSV 和运行日志中记录实际半径。默认旧清单仍兼容 30 张图片。更新了控制流测试，并增加 `parnv-MC/launch_rednet_50.py` 准备与启动脚本。没有进一步修改核心 CEGAR 算法。

本地 `D:/p_workplace/Prune` 与服务器 `/home/test/yiyuanchun/Prune-v2` 的 760 个源码、配置和文档文件通过逐文件 SHA-256 校验。需要上传的三个适配文件已上传；覆盖前的服务器文件保存在 `results/rednet_50_sync_20260913/`。同步清单位于本次实验目录的 `audit/local_sync_sha256.json`。

## 实验配置

- 服务器：`test@192.168.0.239`；环境：`alpha-beta-crown`。
- 流程：仅 REDNet + CEGAR。
- 模型：`data/models/mnist/mnist_fc_relu_64x3.onnx`，结构 784→64→64→64→10。
- 输入：沿用已校验的 ONNX 对应 NNet，模型和数据哈希校验通过。
- 数据：MNIST train，索引 **0–49**，共 50 张；全部由原始模型正确分类，无跳过样本。
- 半径：0.02 和 0.03，使用完全相同的 50 个样本，分别执行 50 次性质验证。
- 每次 Marabou 正式求解超时：**3600 秒**。沿用此前配置，不额外设置整个样本的总超时。
- 默认 PGD：40 步、5 个起点、seed=0；批量精化额外恢复最多 3 个 Merge；总精化步数不限制。
- REDNet 在每张图片的实际输入盒内执行 128 次数值等价性检查，容差 `1e-8`，通过后才进入 CEGAR。
- 两个独立后台进程分别绑定 CPU 8–11 与 12–15；数值库线程数为 1。

本次独立实验根目录：

```text
/home/test/yiyuanchun/Prune-v2/results/rednet_50_20260913
```

以下路径均相对于这个目录：

| 内容 | ε=0.02 | ε=0.03 |
|---|---|---|
| 样本及配置 | `samples_eps002.json` | `samples_eps003.json` |
| 正式输出 | `full_eps002/` | `full_eps003/` |
| 正式 CSV | `full_eps002/b3_rednet_narv_eps002.csv` | `full_eps003/b3_rednet_narv_eps003.csv` |
| 正式日志 | `full_eps002.log` | `full_eps003.log` |

公共样本清单是 `sample_indices.json`；对应性质文件位于 `properties/`。旧的 30 张清单和旧实验结果保留。`audit/source/` 保存 268 个 Python 源文件快照，`audit/source_hashes.json` 保存哈希，启动器会检查代码没有在准备后变化。

## 检查与运行状态

相关测试：**31 passed**，包括提前 PGD 找到真实反例时不调用 Marabou、原有批量精化与预算边界测试、REDNet 回归测试和 CIFAR 输入性质测试。测试日志为 `audit/tests.log`。

已先为两个半径分别启动样本 0 的小规模验证，并附加直接 Marabou 求解交叉核对；附加求解不计入生产流程耗时与调用次数。

两项小规模验证均已完成，且直接求解结论一致：

| 半径 | 样本 0 结论 | 生产流程 Marabou 调用 | PGD 调用 | 生产流程总耗时 | 附加直接求解 |
|---|---|---:|---:|---:|---|
| 0.02 | VERIFIED | 1 | 2 | 159.583 秒 | UNSAT |
| 0.03 | UNSAFE | 0 | 2 | 145.870 秒 | SAT |

0.03 的真实反例来自 PGD。独立复核了输入盒、原始 NNet 和原始 ONNX：样本原标签为 5，反例被两种格式的原始模型均分类为 3；最大输入改变量约 0.027，ONNX 的非目标类别优势约 0.759605。复核结果位于 `audit/eps003_original_counterexample_check.json`。这是单个样本的反例，不代表其余 49 张的结论。

正式实验已于 **2026-09-13 17:28:34 +0800** 启动，并已确认两个进程均在处理样本 0：

| 半径 | PID | 样本数 | 流程 |
|---|---:|---:|---|
| 0.02 | 2927982 | 50 | REDNet + 新 CEGAR |
| 0.03 | 2927983 | 50 | REDNet + 新 CEGAR |

已读取两个任务的 `run_config.json`，确认实际配置为 REDNet、索引 0–49、对应半径和 3600 秒求解超时。正式实验仍在后台运行，尚未完成 100 次性质验证；按此前约定，确认启动后不等待所有样本结束。

## 重现命令

在服务器项目目录、激活同一环境后执行以下命令。准备阶段要求使用一个尚不存在的新目录；后续阶段使用相同目录，避免覆盖既有结果。

```bash
python parnv-MC/launch_rednet_50.py prepare --run-root results/<新的实验目录> --source-commit 2bba1074c343809ef61e1f7f5d6e26c5d3e6a227
PYTHONPATH=parnv-MC python -m pytest -q parnv-MC/tests/test_refinement/test_batch_pgd.py parnv-MC/tests/test_pre_process/test_b3_rednet.py parnv-MC/tests/test_utils/test_cifar10_property_utils.py > results/<新的实验目录>/audit/tests.log 2>&1
python parnv-MC/launch_rednet_50.py smoke --run-root results/<新的实验目录>
# 等两个 smoke 日志均出现 B3 COMPLETE，再执行：
python parnv-MC/launch_rednet_50.py full --run-root results/<新的实验目录>
```

正式任务的启动命令、PID 和路径将保存在 `full_launch.json`。各输出目录的 `progress.json` 记录当前样本；CSV 按已完成样本逐行写入，完整结束时日志输出 `B3 COMPLETE`。
