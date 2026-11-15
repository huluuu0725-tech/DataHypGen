# DataHypGen Benchmark

评估系统能否从地理空间数据出发，提出有依据、具有研究深度且可执行的研究方案。

## 任务与评价

正式任务以 `dataset/task_groups.json` 为准，共101个：单数据集49个、双数据集32个、三及以上数据集20个。

三个评价维度分别报告：

- **Alignment**：方案与正式参考研究方向的对应程度。
- **Depth**：完整研究方案的研究深度，按1–10档评审。
- **ResourceCheck**：方案需要的数据和工具是否可用。

## 目录

| 路径 | 用途 |
|---|---|
| `dataset/task_inputs/` | 101个正式任务的生成输入 |
| `dataset/paper_ideas/`、`dataset/seed_papers.json` | 评测侧原论文资料 |
| `reference_sets/<group>/` | 正式任务参考方向与证据 |
| `reference_sets/knowledge_graph/` | 参考检索所用图谱与产品关系 |
| `protocol/task_input_contract.md` | 任务输入约束 |
| `protocol/depth_review_prompt_v1.md`、`protocol/depth_rubric_v1.json` | 实际使用的Depth评审提示词与量表 |
| `scripts/` | 标准化、评测和汇总等阶段入口 |
| `resource_check/` | 评测侧资源需求提取、资源检查与原生方案入口实现 |
| `core/` | 准入、任务布局、Depth、语义检索与资源检查公共实现 |
| `tasks/` | 五个真实代码验证扩展任务的输入及通用实验说明 |
| `run.py` | 仓库根目录运行入口 |

生成阶段只读取任务输入。原论文资料和参考集用于评测。

## 使用

在仓库根目录、已安装项目依赖的Python环境中运行：

```sh
python benchmark/run.py list
python benchmark/run.py summary --dry-run
```

`--dry-run`显示将执行的命令；去掉后执行对应阶段。其他参数传给对应脚本。

入口调用当前 `benchmark/scripts/`，公共实现位于 `benchmark/core/`，评测协议位于 `benchmark/protocol/`。任务、参考集与默认系统输出均定位到当前 `benchmark/`。

运行仍依赖仓库中的 `datahypgen/`、`config/` 及项目第三方依赖；资源检查位于 `benchmark/resource_check/`。不能只复制 `benchmark/` 单独运行。

统一生成入口为 `python launch_discovery.py --benchmark all`。将 `all` 换为分组名或规范任务 ID 可缩小范围。文献缓存运行时生成到 `benchmark/cache/filtered_literature.json`，不依赖历史实验缓存。

当前评测与汇总的系统选项仅支持 `ours`；`--systems all` 同样只选择 ours。缺失资源检查配置会直接报错，不使用其他系统配置代替。

## 配置与复现前提

统一生成入口默认使用根目录 `config/debug_config.yaml`，可通过 `--config` 指定其他配置。普通入口输出到 `results/<task_name>/`；benchmark 输出到 `benchmark/system_outputs/ours/<group>/<task_id>/`。

评测默认读取根目录 `config/debug_config.yaml`，资源检查使用其中的模型、工具和 Agent 参数，Alignment 使用 `semantic_retrieval` 节。

```sh
python benchmark/run.py evaluate -- --axes all --systems ours
python benchmark/run.py summary
```

可通过 `--resource-check-config` 和 `--retrieval-config` 显式覆盖配置路径。模型凭据和服务地址通过环境变量提供。
