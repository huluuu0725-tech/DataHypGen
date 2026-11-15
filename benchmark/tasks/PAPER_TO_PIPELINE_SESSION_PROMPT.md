# 新会话启动提示词：从一篇论文构建任务，走 DataHypGen 管线并完成生成结果评估

> 把本提示词整段作为新会话的首条消息，替换 `{}` 占位符后即可开工。
> 本提示词编码了 D-002 案例（postfire_debris_flow_initiation）全程验证过的工作流与全部踩坑教训。

---

## 你的任务

给定一篇**有配套数据与代码**的论文，完成三件事：
1. 把论文的伴生数据构建为规范任务输入（prompt.json）；
2. 用该任务跑通 DataHypGen 管线，生成并成熟化研究 idea；
3. 对生成的 idea 完成源码级评估：与原论文原配置做增益比较，按预登记证伪条件判定成立/证伪。

## 工作区与环境

- 仓库根：`/Users/swx/Developer/DataHypGen`（**所有命令从仓库根执行**；launch_discovery 自动 `load_dotenv()` 读 `.env`：QWEN_API_KEY / OPENAI_API_KEY / GEE_API_KEY 等）
- 运行环境：`.venv-pipeline/bin/python`
- 本维度目录：`benchmark/tasks/<任务名>/`（任务名 = 描述性英文短横线名，如 `postfire_debris_flow_initiation`；内部含 `prompt.json` 与 `validation/`）
- 数据目录：`datasets/<任务名>/`
- 方法论文件：同目录 `IDEA_GAIN_COMPARISON_PROMPT.md`（增益比较三表与判定规则，第三步必读）
- 参照案例：D-002 / `postfire_debris_flow_initiation`（任务、验证产物、会话轨迹齐全，可对照）

## 第零步：从论文清单选取论文

论文主清单与全部检索产物在 **`<本地论文与检索材料目录>/`**：

| 文件 | 用途 |
|---|---|
| `candidate_registry.csv` | **主清单（112 篇候选，21 列）**：candidate_id（A/B/C/D 四方向 + 序号）、title、DOI、status（审计漏斗：DISCOVERED→…→REPRODUCED）、code_url、data_url、许可证、audit_card_path、exclusion_reason |
| `exclusion_log.csv` | 23 条已排除记录——**选取前先对照，禁止重选已排除论文** |
| `dataset_families.csv` | 21 个数据集家族——窗口外外推时按家族找伴生数据（如 D-002 ↔ USGS inventory 10.5066/P1R9VXC4） |
| `search_log.csv` / `input_platform_mapping.csv` / `journal_whitelist.csv` | 检索过程、平台映射、期刊参考 |
| `reports/final_candidate_list_20261031.md`、`reports/full_survey_20261031.md` | 人读版终选清单与完整调研 |
| `wildfire_benchmark/tasks/`（审计卡）/ `evaluator_private/` | 已审论文的审计卡与 idea 存档 |

选取规则：
1. **只选 status 尚未进入 REPRODUCED/ACCEPTED 且不在 exclusion_log 里的条目**；优先 ASSET_PASS 以上的（已确认有配套数据+代码）；
2. 硬性准入：**有可下载的伴生数据与代码、零邮件依赖**——不满足则跳过（无资产即换论文）；
3. 若该条目尚无审计卡，先按 `specification.md` 的漏斗补审计（ASSET_PASS：数据可下载、许可证、规模、代码可复现性核验），再进入第一步；
4. **登记义务**：选定后在 `candidate_registry.csv` 该行回填 `task_summary`、本维度任务目录名（`benchmark/tasks/<任务名>/`）与状态推进；candidate_id 与任务目录名的映射以 registry 为准（目录名进论文，编号进登记表）。

## 第一步：论文 → 任务构建

1. **资产获取**：论文全文（有 Cloudflare 的用浏览器通道取，存 `validation/paper_fulltext_<date>.txt`）；伴生数据/代码（Zenodo/GitHub，快照落 `<本地论文配套资产目录>/`）。**零邮件政策**：需要向作者索要数据的一律换论文。**无配套数据/代码的论文不进本维度**。
2. **数据落位**：`datasets/<任务名>/`（CSV 优先；xlsx 保留原样）。
3. **prompt.json 必须包含**（对齐仓库任务输入密度，参照 D-002 实例）：

| 字段 | 内容 | 为什么 |
|---|---|---|
| `system` | 一句通用角色设定 | 不做方向暗示 |
| `region_name` | 数据集一句话定位 | |
| `data_description.dataset_name / spatial_coverage / temporal_coverage / data_type / data_format / records / total_samples / data_type_numeric` | 规模与形态事实 | |
| `table_content.label` | 标签定义 + 类平衡（正例数/正率） | |
| `table_content.column_keys` | 全部列名清单 | |
| `table_content.column_groups` | 列的语义分组 | |
| `unit_of_measurement` | 每列单位与量纲关系（含"嵌套峰值/比例"这类结构性关系） | 防止把嵌套通道当独立变量 |
| `important_note` | **结构性事实清单**：时间窗边界（含哪些窗口为空）、哨兵值与缺失模式、采样限制（如"仅第一年"）、来源数据的已知局限 | 让 affordance 检查在生成期就能咬住不可测主张 |
| `column_profiles` | 逐列：type / missing / min / p50 / max（数值）或 cardinality / top_values（分类） | 方向性主张的分组 bin 必须落在真实范围内 |
| `dataset_facts` | 采样结构：每火/每年事件数、流域数、正例分布、正例覆盖范围 | |
| `data_source / reference` | 数据出处与 DOI（区分"官方发布"与"作者伴生汇编"） | |
| `analytical_capabilities` | 表能支撑/不能支撑的分析类型 | |
| `data_path` | `./datasets/<任务名>/`（**仓库根相对**，任务目录搬家不影响） | |

4. **中性红线**：只描述事实，**零假设注入**——不写"值得研究 X"、不暗示任何方向。异常值/尖峰只报计数与位置，不做解读。

## 第二步：调用管线

```bash
cd /Users/swx/Developer/DataHypGen
nohup .venv-pipeline/bin/python launch_discovery.py \
  --task benchmark/tasks/<任务名> \
  --config config/debug_config.yaml \
  > /tmp/<任务名>_pipeline.log 2>&1
```

- `--task` 传**已存在的目录路径**（含 `/` 即按路径解析，task_name 取目录 basename）；`--config config/debug_config.yaml` 是全量系统档（`enable_idea_maturation: true`、10 agent 全 qwen3.7-flash、idea_evaluation 走 `default_provider: openai` = gpt-5.6-sol 跨模型评审、调整后六轴 FV 权值）。
- 阶段流：profiler → survey（arXiv/OpenAlex，Wiley 403/PANGAEA 500 为**已知非致命**）→ domain_knowledge → plan（idea_generation → **成熟化**：苏格拉底双树 + 轮内探针 + 终检 → step_plan）→ resources_check + plan refinement → idea_evaluation。
- 时长参考：总计约 **35-47 分钟**；planning 块（生成+成熟化+探针+终检+step_plan）约 18-32 分钟，是方差主体（轮数由 FV 达标/停滞决定）。
- 结果落点：`results/<任务名>/traj_session_*/traj_session_*.json` + `discovery_summary.json`。
- 完成判据：`EXIT=0`、无 `Traceback`、日志尾部 `All done!`。
- 监控要点：`Idea maturation round audit`（每轮质询/FV）、探针挂载、`final_empirical_check`。

## 第三步：生成结果评估

1. **轨迹提取**（traj JSON 字段地图）：
   - `application_records[*]`：`idea_candidate`（终版主张字段，含终检修复后的方向核）、`idea_evaluation.aggregates`（quality/validation/extended/min-core）、`status`（`failed_before_evaluation` 指计划层失败，idea 仍已评审）；
   - `planning_audit.idea_maturation_traces[*]`：`initial/best_selection`（FV 轨迹）、`rounds[*].questioning_findings[*].micro_empirical_check`（探针统计）、`final_empirical_check`（终检：probe_attached / contradicted / repair_accepted / direction_source）。
2. **LLM 评审分只作排序与筛选**，永远不是真值——实测校准三轮：quality 分不预测源码验证结果。
3. **源码验证流程**（对每个 idea）：
   - 冻结原配置：论文过滤规则、特征构造、模型超参、折协议、阈值、指标**逐位复现**，三层验收（过滤行数精确、披露系数逐位、指标 ±0.005）通过后才算基线；
   - idea = 基线上的**最小增量**（禁止混入任何其他改动，否则 Δ 无法归因——两个方向都会出错）；
   - 双协议配对：P1 事件随机 4 折×10 + P2 按火/流域分组 4 折×10，同折喂基线与 idea 版，逐折配对 ΔTS/ΔTSS/ΔAUC；
   - 判定按 idea 自己的 `falsification_condition`（预登记先行）；效应主张用 Wald z（按 idea 声称的控制集），分布主张用 MWU/Fisher；**原始显著不算数，控制后消失即证伪**；
   - 需要外部数据（GEE/家族数据集）的主张：GEE 服务账号可用；窗口外外推必须换家族数据并单独标注，禁止跨表移植数值结论。
4. **增益比较与输出**：读同目录 `IDEA_GAIN_COMPARISON_PROMPT.md`，产出三表（增益总表 / Δ指标表每行标注基线 / 新信息清单含对原论文结论的修正含义）。

## 检查清单（易踩坑，逐条过）

- [ ] prompt.json 从新位置加载成功、`data_path` 从仓库根可解析；
- [ ] 论文全文与伴生数据均无邮件依赖；
- [ ] 运行日志无 Traceback、EXIT=0；
- [ ] 成熟化轮内探针有挂载、`final_empirical_check` 已执行（无条件终检，逐 idea 记录）；
- [ ] 每条存续方向性主张都能指出它的统计出处（探针/终检），否则不得进入结论；
- [ ] 结论方向句与统计逐字一致（禁止主语互换换皮）；
- [ ] 双协议都跑、配对 Δ 逐折呈现、每行 Δ 标注基线；
- [ ] 结构性边界（时间窗/哨兵/单表限制）写进每条结论；
- [ ] 历史轮结论未移植；LLM 分未当真值。

## 禁止事项

- 零邮件；无资产即换论文；大资产不落本机盘（快照只留代码与小文件）；
- 禁止 subagent / ultra 档（除非当场向用户说明并获准）；
- 禁止破坏性 git 命令与未经确认的删除；
- 禁止把效应估计/描述统计读成"高好低坏"的性能分；
- 实验结果默认对话汇报，不主动落文档（用户明确要求才写）。

运行前须指定实际存在的论文、配套资产和验证目录；文中的占位路径由使用者提供，不代表仓库已附带这些材料。
