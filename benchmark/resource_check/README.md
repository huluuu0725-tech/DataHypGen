# 评测侧资源检查

本目录提供 benchmark 和 baseline 共用的资源需求提取、数据检索、工具证据与科学约束核对实现。

| 文件 | 职责 |
|---|---|
| `checker.py` | 共享资源检查器 `ExperimentResourceChecker` |
| `semantics.py` | 显式资源需求提取和证据契约 |
| `native.py` | 原生自然语言方案的提取及评测适配 |
| `cli.py` | 命令行参数、输入校验、配置加载及结果写入 |
| `config.yaml` | 提取模型与配置来源 |

在仓库根目录执行：

```sh
python benchmark/run.py native-resource-check -- --input <方案文件或批次目录> --dry-run
python benchmark/run.py native-resource-check -- --input <方案文件或批次目录> --extract-only
python benchmark/run.py native-resource-check -- --input <方案文件或批次目录>
```

原生方案入口支持 DeepScientist、Denario、IDEAgent。`--dry-run` 只做输入预检；`--extract-only` 调用配置模型提取资源需求；完整检查会调用模型、检索和外部资源探查。默认扫描 `benchmark/system_outputs/`，正式评测应显式指定批次。

配置从仓库根目录解析：统一读取 `config/debug_config.yaml` 中的模型、工具和资源检查参数。运行需要 `datahypgen/`、能力索引及对应模型、目录检索和数据探查依赖。本目录实现及默认配置文件无需从 `experiments/` 加载；project 兼容调用必须显式提供 `--input`，不再隐式扫描旧实验输出。

只提取方案显式声明的数据、操作、派生产物及约束，不反推隐含输入。资源状态为 satisfied、contradicted、unverified、derived；未找到证据不等于资源不存在。工具证据不证明被评系统已经安装该工具。本模块不执行实验、不重规划方案。

每份 `proposals.json` 同目录输出 `resource_requirements.json` 和 `resource_check_evaluation.json`。源方案保持原样；配置、实现、索引或输入改变后，旧缓存拒绝复用，覆盖需显式 `--overwrite`。
