# 渐进式 Merge 修改与 MNIST 服务器验证

日期：2026-09-15。需求文件：`change_merge_flow.txt`。本次实现限定于 `parnv-MC/`，基于本地提交 `00308879d54f23037704030f2dbbdbee479b1367` 加本次工作区修改。

## 文件和函数

| 文件 | 修改内容 |
|---|---|
| `parnv-MC/core/cegar/progressive_merge.py`（新增） | 独立的渐进式调度、逐层处理、日志和计时 |
| `parnv-MC/core/cegar/raw_cegar.py` | 接入调度；初始化所有隐藏层的 unknown 标签；跨层撤销的 split 检查点；LP 子计时；结果统计 |
| `parnv-MC/core/utils/marabou_query_utils.py` | 修复单隐藏层及无隐藏层网络输出变量错误追加 `_b` 的问题 |
| `parnv-MC/b3_mnist.py` | CSV 增加渐进式统计，运行配置标记 `progressive-merge-initial-pgd-v3` |
| `parnv-MC/launch_rednet_50.py` | 实验名称标记为 `rednet_50_progressive_merge` |
| `parnv-MC/tests/test_refinement/test_progressive_merge.py`（新增） | 新流程、真实 LP / split / undo、真实 Marabou 回归测试 |
| `parnv-MC/tests/test_refinement/test_batch_pgd.py` | 控制器测试通过独立调度接口注入测试状态，保留既有 PGD / 精化断言 |

新增主要函数：`clone_baseline_state`、`prepare_layer_for_merge`、`merge_confirmed_layer`、`explore_layer_with_random_tests`、`run_progressive_equivalence_preprocessing`。内部 `_merge_layer` 复用同一套 CROWN、选对、LP Merge 实现；`ProgressiveResult` 返回当前状态、全轮报告和统计。旧的 `merge_last_two_hidden_layers` 保留为最多处理两层的兼容入口，主流程使用新调度器处理所有隐藏层。

## 控制流

原流程固定处理最后两个隐藏层，按层分类或拆分后连续 Merge 到拆分前参考大小的 50%；中途随机检测到违反性质时仍可能继续 Merge。

新流程在 REDNet 恒不激活剪枝、恒激活重构及 CEGAR 的恒不激活剪枝之后，只建立一次干净的 `baseline_state`。从最后隐藏层开始，每轮只新增一个探索层；下一轮深复制原始 baseline，再从输出端向前重放已完成层。本轮目标始终为 `ceil(0.5 * baseline 中该层的大小)`，不以 split 后增大的大小为基准。

已完成层重做必要分类、split 和 Merge，达到目标期间不执行 random test。新探索层每成功完成两次 Merge 检测一次；若最后只需一次即可达到目标，也立即检测。检测到 `has_violation=True` 就停止当前层和外层扩展，保留当前轮状态，交给既有 PGD / CEGAR / 精化 / 正式求解流程。随机检测本身不会输出 UNSAFE，也不构成安全证明。

分类在进入具体层的 Merge 函数后才执行。未访问的更前层只保留 unknown 标签；不会预先 split。已经达到目标的层也跳过分类。后继层缺少有效单调性标签时停止扩展并记录原因，避免猜测标签。没有合法同标签 pair 时记录 `no_merge_pair`，不强制合并。

每轮深复制权重、偏置、IDs、标签、mapping 和历史；前轮 Merge / refinement 记录不进入后轮。只把最后实际保留的一轮 Merge 栈交给精化。重放重新计算 CROWN 和 LP，没有引入跨轮状态或数值缓存。

```text
baseline = state(less_neuron_network)  # 仅构造一次，后续只读
confirmed = []
for exploration_layer in 隐藏层从后向前:
    state = deepcopy(baseline)
    for layer in confirmed:
        按需 classify/split，Merge 到 target，不做 random test
        无合法 pair 时保留 state 并转入 CEGAR
    按需 classify/split exploration_layer
    while 当前大小 > ceil(0.5 * baseline 参考大小):
        选择同标签 pair 并执行一次 LP Merge
        无合法 pair 时保留 state 并转入 CEGAR
        if 本层成功 Merge 次数为偶数 or 已达到 target:
            if random_test.has_violation:
                保留 state，结束所有扩展，转入 CEGAR
    confirmed.append(exploration_layer)
使用最后一轮 state，继续 PGD / CEGAR / refinement / Marabou
```

## 日志与计时

`cegar_log.json` 增加 `progressive_round_reports` 和 `progressive_statistics`。最终结果 JSON 和 B3 CSV 增加：

- `progressive_rounds`、`progressive_confirmed_layers`、`progressive_stop_layer`、`progressive_stop_reason`。
- `progressive_preprocessing_time_seconds`、`inc_dec_preprocessing_time_seconds`、`merge_total_time_seconds`。
- `merge_crown_time_seconds`、`merge_pair_selection_time_seconds`、`merge_lp_time_seconds`、`merge_random_test_time_seconds`。
- `progressive_round_rebuild_time_seconds`、`network_build_time_seconds`。

逐轮、逐层报告含 `round_index`、`exploration_layer`、`confirmed_layer_count`、`layer_index`、`role`、`size_before_preprocessing`、`size_after_inc_dec_split`、`target_size`、`merge_count`、`random_test_count`、`random_violation_found`、`final_layer_size`；随机检测另记录 `after_merge_count`，Merge 记录带轮次与角色。初始 CROWN 已确定结果而未运行渐进流程时，新计数和计时为 0，停止原因是 `not_started`。

计时累计包含所有尝试轮，包含最终没有保留的前轮。`merge_total` 包含循环内 CROWN、选对、LP、网络构建和随机检测，后五项属于其子计时，不能再次与它相加。分类和轮重建单独计时；所有新增计时均属于原 CEGAR 总时间内部。原有 `cegar_time_seconds`、`crown_time_seconds`、`cegar_crown_time_seconds`、`pgd_time_seconds` 的定义不变。

## 正确性与 soundness 检查

保留单方向只标记、混合方向才 split、只有相同 inc/dec 标签才能 Merge 的约束，LP 约束和 outgoing edge 拆分公式不变。保留 REDNet、CROWN、初始 PGD、批量精化、原始网络反例复核和 Marabou 正式验证。

发现并处理了两个与扩展范围直接相关的问题：

1. 更前层的 split 会改变已合并后层权重矩阵的列数。直接撤销后层旧 Merge 会产生维度不一致。现在分类前保存带 Merge 栈深度的等价预处理检查点；精化跨越该边界时先还原相应 split，再撤销更早的 Merge。等价 split 还原不额外消耗 Merge 撤销预算，mapping 记录还原历史。
2. 扩展可能到达第一隐藏层。原有 LP 失败后的逐分量回退依赖前驱非负；若输入下界存在负数且发生该回退，现在在修改网络之前拒绝，避免给出无依据的安全结论。正常 LP 保持原有逻辑；本次 MNIST 输入盒裁剪至 [0,1]，不触发此保护。

另修复浅层网络正式查询对输出节点错误使用隐藏层变量后缀的问题，并用有/无隐藏层的 SAT、UNSAT 查询验证。测试也遇到十进制浮点边界造成严格反例复核拒绝的情况；最终接口测试使用精确二进制数，生产反例判定没有放宽容差。现有浮点 CROWN / LP / Marabou 数值容差及 split 的微小边权容差仍是继承的数值假设，本次测试不是整体形式化 soundness 证明。

## 测试

服务器 `alpha-beta-crown` 环境：**47 passed，1 个可选 TensorFlow parser 缺失警告，2.03 秒**。

覆盖需求列出的 10 类场景，另覆盖无隐藏层、目标已达到、无合法 pair、计时、跨层 split 撤销、含负输入时的 LP 回退保护，以及真实 Marabou SAT / UNSAT 和候选复核。随机检测报告 violation 的集成测试仍由真实 Marabou 得到 VERIFIED，验证随机结果只控制停止 Merge。

运行文件：新 `test_progressive_merge.py`、既有 `test_batch_pgd.py`、`test_b3_rednet.py`、`test_cifar10_property_utils.py`，以及 `test_marabou_query_utils.py` 中的 `test_get_query_1` 和 `test_handle_adversarial_matches_raw_sum_relu_margin`。这是针对性回归集合，未声称仓库所有历史测试均通过。修改文件语法及 `git diff --check` 通过；未改动的 Plane 目录仍含 Python 2 工具。

## 服务器部署和实验

- 服务器：`gpu@192.168.0.44:22`。
- 项目：`/home/gpu/yyc_projects/parnv-v2`。
- 环境：`conda activate alpha-beta-crown`。
- 已同步并 SHA-256 校验 831 个源码、配置和需求文件；保留服务器数据、已有结果及额外历史文件。
- 部署前备份和历次测试日志：`results/progressive_merge_sync_20260915/`；最终同步哈希表 `source_hashes.json`。
- 最终源码包 SHA-256：`3132f7b7447d981b9f5b8e4970560024d22b0b96b857a6ef2c65619c55c7efea`。
- 实验根目录：`results/progressive_merge_50_gpu44_20260915/`。

沿用同一 MNIST FC ReLU 784→64→64→64→10 模型、同一 MNIST train 数据与 0–49 共 50 个正确分类样本。两个半径为 0.02、0.03；仅运行 REDNet + CEGAR。每次 Marabou 求解超时 3600 秒，PGD 40 步、5 个起点、seed=0，伪反例消失后额外恢复最多 3 个 Merge。两组使用 CPU 8–11、12–15，数值库各 1 线程。REDNet 仍按每样本 128 个点执行内存及序列化等价性检查。

准备时使用 `--base-manifest results/gpu44_sync_20260913/experiment_samples_gpu44.json`，校验旧实验模型与数据哈希；运行时使用新生成的两份 `samples_eps002.json` / `samples_eps003.json`。源码快照及实验参数位于 `audit/` 和对应配置文件。

两项样本 0 试跑均已完成，附加直接 Marabou 查询与流程结果一致。附加直接查询的耗时、调用数不计入下表生产流程统计。

| 半径 | 结果 | Marabou 调用 | PGD 调用 | 精化撤销数 | CEGAR 耗时 | 全流程耗时 | 直接求解 |
|---|---|---:|---:|---:|---:|---:|---|
| 0.02 | VERIFIED | 1 | 2 | 5 | 36.370 秒 | 39.674 秒 | UNSAT |
| 0.03 | UNSAFE | 0 | 2 | 2 | 4.634 秒 | 8.036 秒 | SAT |

两者均执行一轮，只处理最后隐藏层（层 3），停止原因均为 `random_violation`，confirmed 层数均为 0：

| 半径 | 分类前大小 | split 后大小 | 目标大小 | 成功 Merge 次数 | 随机检测次数 | 停止时大小 |
|---|---:|---:|---:|---:|---:|---:|
| 0.02 | 24 | 35 | 12 | 12 | 6 | 23 |
| 0.03 | 32 | 49 | 16 | 2 | 1 | 47 |

对应渐进式预处理总时间分别为 8.931 秒、1.574 秒。日志确认检测后立即停止，未强行达到目标，也未进入前两层。0.03 的 PGD 反例通过输入盒、原始 NNet、原始 ONNX 复核：原标签 5，反例预测 3，最大扰动约 0.027，ONNX 类别优势约 0.759605；结果见 `audit/eps003_original_counterexample_check.json`。

这些只是单样本测量，不能据此断言整个 50 样本批次的加速比例。正式任务的启动信息见 `full_launch.json`，逐样本进度见 `full_eps002/progress.json` 和 `full_eps003/progress.json`。

正式后台任务已于 **2026-09-15 10:30:32 +0800** 启动，并核实两个进程存活、实际配置包含 50 个样本和 3600 秒求解超时，且均启用新流程：

| 半径 | PID | 样本数 | CPU |
|---|---:|---:|---|
| 0.02 | 1701942 | 50 | 8–11 |
| 0.03 | 1701943 | 50 | 12–15 |

交付时正式批次仍在运行，尚未得到完整 100 次性质验证的最终汇总。沿用此前约定，确认后台任务正常启动后不等待所有样本结束。

核心运行命令如下。正式运行已经由启动器安排为独立后台进程；查看现有任务时无需再次执行，避免重复计算。

```bash
cd /home/gpu/yyc_projects/parnv-v2
conda activate alpha-beta-crown
python -u parnv-MC/b3_mnist.py --route rednet \
  --samples results/progressive_merge_50_gpu44_20260915/samples_eps002.json \
  --output results/progressive_merge_50_gpu44_20260915/full_eps002
python -u parnv-MC/b3_mnist.py --route rednet \
  --samples results/progressive_merge_50_gpu44_20260915/samples_eps003.json \
  --output results/progressive_merge_50_gpu44_20260915/full_eps003
```

两份正式 CSV 分别是 `full_eps002/b3_rednet_narv_eps002.csv` 和 `full_eps003/b3_rednet_narv_eps003.csv`。查看日志：

```bash
tail -f /home/gpu/yyc_projects/parnv-v2/results/progressive_merge_50_gpu44_20260915/full_eps002.log
tail -f /home/gpu/yyc_projects/parnv-v2/results/progressive_merge_50_gpu44_20260915/full_eps003.log
```
