# MNIST 验证：只保存 CSV

`parnv-MC/b3_mnist.py` 默认只在 `--output` 目录保存汇总 CSV。保留当前全部结果列，包括你提供的 CSV 中的结果、耗时、剪枝及等价性统计，以及当前代码已有的 epsilon、PGD 和渐进式 Merge 统计。

REDNet、序列化等价性检查、CROWN、渐进式 Merge、PGD、精化、反例复核和 Marabou 求解流程不变。每个样本仍使用相同配置和随机种子；减少文件写入后，测得的运行时间可能下降。

默认不保存逐样本 JSON、mapping、Merge/精化日志、中间及最终 `.nnet`、Marabou `.query`、`run_config.json`、`progress.json`。REDNet 为序列化检查所需的 `.nnet` 放在系统临时目录，在当前样本结束或 Python 异常退出该样本时清理。强制杀进程或断电可能留下临时目录。内存中用于精化的 Merge 历史仍保留。

终端保留开始、逐样本进度和结束信息；异常及 traceback 仍写到 stderr。每个完成的样本立即写入并刷新 CSV，求解异常会写入 ERROR 行后停止，已完成行不会丢失。已有同名 CSV 不会被覆盖，重跑请换 `--output` 目录。

## 沿用上一轮的 50 样本配置

以下命令在已经配置好模型、数据和 `alpha-beta-crown` 的运行机器上执行。此次仅修改本地代码，没有同步或操作服务器；如果在服务器运行，需要先自行同步本次本地改动。不要只复制 `b3_mnist.py`，新的 `core/utils/experiment_output.py` 及关联修改也需要带上。

复用服务器上已有的两份配置，仍为同一批 50 个样本，半径分别 0.02、0.03，每次 Marabou 超时 3600 秒：

```bash
conda activate alpha-beta-crown
cd /home/gpu/yyc_projects/parnv-v2

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''

python -u parnv-MC/b3_mnist.py --route rednet \
  --samples results/progressive_merge_50_gpu44_20260915/samples_eps002.json \
  --output results/mnist_csv

python -u parnv-MC/b3_mnist.py --route rednet \
  --samples results/progressive_merge_50_gpu44_20260915/samples_eps003.json \
  --output results/mnist_csv
```

上述两条验证命令顺序执行。输出目录只新增：

```text
results/mnist_csv/b3_rednet_narv_eps002.csv
results/mnist_csv/b3_rednet_narv_eps003.csv
```

如果需要创建新的 50 样本清单，可先执行以下准备命令，再把上面 `--samples` 的目录换成 `results/mnist_csv_config`：

```bash
python parnv-MC/launch_rednet_50.py prepare \
  --run-root results/mnist_csv_config \
  --base-manifest results/gpu44_sync_20260913/experiment_samples_gpu44.json
```

准备阶段保留运行所需的样本清单和源码哈希，但默认不再复制整份源码或写出 100 份性质文件。模型和数据位于其他机器时，需要使用该机器路径有效的 `--base-manifest`；不要直接使用含旧服务器绝对路径的根目录 `experiment_samples.json`。

## 调试开关与原有启动器

给 `b3_mnist.py` 的命令追加 `--save-artifacts` 可恢复详细输出。日常只需要 CSV 时不要加这个参数。

原来的 `launch_rednet_50.py smoke/full` 也兼容：smoke 为直接求解交叉检查自动保留详细文件，full 默认只保存逐样本汇总 CSV；启动器仍保留进程信息及控制台日志。若需要严格只保留 CSV，使用上面的 `b3_mnist.py` 入口即可。

## 本地检查

5 项输出回归测试通过，覆盖轻量/详细模式的结果与调用一致性、临时文件正常/异常清理、ERROR 行、禁用查询保存（含环境变量路径）和标准错误输出。测试使用模拟求解调用，本机缺少 PyTorch / Marabou，未重新运行数值验证。

```text
python parnv-MC/tests/test_utils/test_experiment_output.py -v
```
