# Batch Refinement + PGD 修改与服务器验证记录

日期：2026-09-13。需求：`reduce_iteration.txt`，并沿用 `parnv.txt` 的同样本、双后台流程实验要求。

## 修改范围

- `parnv-MC/core/cegar/raw_cegar.py`：修改 `refine_by_undo_merge`、`cegar_verify_with_marabou`、反例输入检查、`_finalize` 和命令行参数。
- `parnv-acasxu/core/cegar/raw_cegar.py`：同步上述策略，保留原有 inc/dec 预处理、等价性检查及 `write_detailed_results` 开关。
- `parnv-MC-plane/core/cegar/raw_cegar.py`：同步策略至 `cegar_verify_with_planet`，兼容入口转发新参数，正式后端仍是 Planet。
- 三个版本的 `core/cegar/pgd_search.py`：相同的性质导向 PGD 实现、配置及命令行选项。
- `parnv-MC/b3_mnist.py`：向新流程传递配置，补充 CSV 字段，并检查正式验证次数与实际观测的 Marabou 查询调用数一致。
- `parnv-MC/launch_reduce_iteration.py`：先检查新 smoke 结果，再启动使用独立目录的两条正式流程。
- `parnv-MC/tests/test_refinement/test_batch_pgd.py`：同一套测试分别加载三个版本执行。

本轮未改动 CROWN、Merge 的 LP、`undo_last_merge` 的快照恢复规则或 REDNet 算法；工作目录中 REDNet 的修改来自上一轮 B3 任务。未改动用户的 IDE 配置。

## 执行逻辑

1. 保留 CROWN 预检查及原有预处理、剪枝与 Merge。
2. 正式求解器 UNSAT 则 VERIFIED；SAT 的输入必须在原始网络上检查，真实违反则 UNSAFE。
3. 对伪反例按栈逆序 undo，直到它消失，再额外 undo 最多 3 个 Merge。两部分都在每次 undo 前检查同一个剩余预算。
4. 只要实际恢复了 Merge，就在更新后的当前网络上执行 PGD。
5. PGD 找不到反例，转正式验证；找到真实反例，返回 UNSAFE；找到伪反例，直接继续批量精化与 PGD，不增加正式验证次数。
6. PGD 伪反例遇到空栈或耗尽预算，退回一次正式验证。正式 SAT 仍是伪反例且没有精化能力，则 UNKNOWN。只有发生实际 undo 才会重新启用 PGD，因此不会出现无进展循环。

## PGD 的目标和边界

PGD 使用当前精化网络的权重与偏置，以 CPU float64 批量前向计算，隐藏层为 ReLU。默认 40 步、5 个起点（一个区间中点、四个固定种子的随机起点），默认每维步长为该输入区间宽度的十分之一；每步投影回性质文件的输入盒。支持 MNIST 的 784 维与 CIFAR-10 的 3072 维。

对于已经变换为类别差值输出的对抗性质，`ge` 目标为 `max_i(output_i - Lower_i)`，`le` 为 `max_i(Upper_i - output_i)`。普通合取性质最大化全部约束裕量的最小值。这里没有使用交叉熵或重新假设分类标签。遇到不支持的性质编码或隐藏激活，跳过搜索并进入正式验证；非法数值或输入盒会明确报错。

候选输入必须通过输入盒和有限值检查，并在项目原有 `speedy_evaluate` 与性质判断函数上重新验证。UNSAFE 必须由原始网络复核；REDNet 路线继续使用原始分类器进行具体反例检查。PGD 失败永远不能生成 VERIFIED。

## 配置和统计

核心函数新增 `extra_refinement_merges=3`、`pgd_config=PGDConfig(...)`。单网络入口和 B3 入口支持：

```text
--extra-refinement-merges 3
--pgd-steps 40 --pgd-restarts 5 --pgd-seed 0
--pgd-step-size <绝对输入步长>
--disable-pgd
```

`iterations`、`cegar_iterations`、`formal_verifier_calls` 只统计正式验证调用；MC/ACAS 的 `marabou_calls` 同步该值。Plane 使用 `planet_calls`，其 `marabou_calls=0`。

新增 `pgd_calls`、`pgd_time_seconds`、`pgd_candidates_found`、`pgd_genuine_counterexamples`、`pgd_spurious_counterexamples`、`batch_refinement_rounds`、`extra_refinement_steps`。`total_refinement_steps` 和 CSV 中的 `refinement_steps` 包括额外恢复步骤。PGD 记录存于 `cegar_log.json` 的 `pgd_searches`，正式查询仍在 `iterations`，查询与网络文件编号仅随正式验证增长。每个 undo 记录标记 `extra_refinement`。

## 测试结果

- 新增 23 个参数化测试在 MC、ACAS、Plane 三个实际模块中分别全部通过。
- 覆盖需求的 10 类场景：CROWN 直接 UNSAT、首次正式 UNSAT、正式真实反例、消除伪反例后额外恢复 3 次、PGD 失败、PGD 真实反例、PGD 伪反例再次精化、剩余 Merge 不足、空栈、零预算及预算截断。
- 检查权重、偏置、节点 ID、标签、mapping、栈与 refinement_log 的恢复；检查正式调用与 PGD 的事件顺序。
- 真实梯度测试覆盖 784/3072 维、`ge`/`le` 两种方向、越界投影、无反例及非法输入。
- MC 加上上一轮 REDNet 的 5 个回归测试，共 28 项通过。
- 全部旧测试收集被 `test_Network.py` 引用已不存在的 `heuristic_abstract` 阻断。排除此文件后执行：33 passed、2 failed。两处失败为旧 `test_get_query_1` 的 `KeyError: y_b` 和旧 `test_get_query_2` 将 `abstract_network` 的 tuple 返回值当成网络。这些失败不涉及本轮修改的函数，未扩展任务范围修改旧接口。
- Python 语法检查及 `git diff --check` 通过。服务器的 TensorFlow parser 缺失警告不影响本次 NNet/ONNX 与 Marabou 验证。

## 服务器实验

服务器：`test@192.168.0.239`，项目：`/home/test/yiyuanchun/Prune-v2`，环境：`alpha-beta-crown`。

模型：`data/models/mnist/mnist_fc_relu_64x3.onnx`，784→64→64→64→10；两路线共享校验过的 NNet。MNIST train 索引 0–29，epsilon=0.02；模型、数据和样本清单哈希沿用 `experiment_samples.json`。单次正式求解超时 3600 秒，总精化预算不限制。baseline 在本轮表示“原有 inactive 剪枝 + 新 CEGAR”；rednet 表示“REDNet + 新 CEGAR”。

根目录：`/home/test/yiyuanchun/Prune-v2/results/reduce_iteration_20260913`。上一轮 `b3_*` 结果保留。

测试日志和经过测试的源文件快照位于新根目录 `audit/`。IDE 配置会自动上传本地修改；已核对两端源码哈希，`auto_synced_snapshot` 表示发现同步后的快照，并非修改前源码。原始源码以本地 Git HEAD 为准。

两条新流程在样本 0 上均返回 VERIFIED。生产流程的统计如下（额外直接求解不计入这些次数与时间）：

| 路线 | 上轮正式调用 | 本轮正式调用 | PGD 次数 / 伪反例数 | 总恢复 / 额外恢复 | 本轮总耗时 |
|---|---:|---:|---:|---:|---:|
| REDNet + CEGAR | 2 | 2 | 1 / 0 | 16 / 3 | 154.045 秒 |
| inactive 剪枝 + CEGAR | 9 | 2 | 4 / 3 | 62 / 9 | 344.147 秒 |

baseline 的 3 个 PGD 伪反例直接进入后续精化，没有调用正式求解器。REDNet 本样本的调用数未减少。上轮样本 0 的总耗时分别为 153.351 秒、367.673 秒；该单样本且有其他服务器负载的比较不能作为整体加速结论。

两条 smoke 的直接求解均为 UNSAT；已逐项核对正式迭代日志数、PGD 日志数、恢复日志数和 CSV 统计一致。

正式任务已于服务器时间 **2026-09-13 12:22:19 +0800** 启动：

| 路线 | PID | CPU 集合 | 输出目录 | 日志 |
|---|---:|---|---|---|
| REDNet + 新 CEGAR | 2557654 | 8–11 | `full_rednet/` | `full_rednet.log` |
| inactive 剪枝 + 新 CEGAR | 2557655 | 12–15 | `full_baseline/` | `full_baseline.log` |

以上目录和日志均相对于 `/home/test/yiyuanchun/Prune-v2/results/reduce_iteration_20260913`。CSV 分别为：

```text
/home/test/yiyuanchun/Prune-v2/results/reduce_iteration_20260913/full_rednet/b3_rednet_narv_eps002.csv
/home/test/yiyuanchun/Prune-v2/results/reduce_iteration_20260913/full_baseline/b3_baseline_par_eps002.csv
```

`full_launch.json` 保存完整启动命令、PID、smoke 结果和源码哈希。各输出目录的 `run_config.json` 保存参数、样本列表和 CPU 亲和性；`progress.json` 保存当前样本进度。两任务均使用索引 0–29，样本清单为 `/home/test/yiyuanchun/Prune-v2/experiment_samples.json`，本地也保留同名文件。

本次结束交互时，正式 30×2 实验仍在后台运行，尚未完成全部样本。遵循 `parnv.txt` 第十四节要求，启动并确认运行后不等待全部结束。后续请以 CSV 的已完成行及日志中的 `B3 COMPLETE` 判断完成状态。

## 适用范围与限制

PGD 是有限步启发式搜索，可能漏掉反例；正确性依赖后续正式验证而非搜索成功率。额外恢复 Merge 可能减少调用次数，也可能扩大单次求解的网络、增加单次耗时，因此不能仅根据次数推断整体加速。浮点运算仍受原有 CROWN、抽象与求解器的数值行为约束。本次为 CPU 验证；两流程使用不同 CPU 集合，但服务器其他负载仍可能影响耗时对比。
