import os.path as osp
import sys
import json
import shutil
from datetime import datetime

from datahypgen.mas.interface import DataGenHypInterface
from datahypgen.mas.agents import plan_utils

EXTERNAL_DATASET_SOURCES = frozenset({
    "PANGAEA",
    "NASA_CMR", "PLANETARY_COMPUTER", "EARTH_SEARCH", "OSM_OVERPASS", "API",
})

# 已删除visualize_hypotheses导入，因为不再需要可视化
# from datahypgen.vis import visualize_hypotheses


# ============================================================================
# Discovery Session Runner
# ============================================================================
class FindingGenerator:
    """Runs the geospatial discovery session through the active MAS pipeline."""
    
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        # work_dir 优先用 output_dir，让 idea maturation audit 文件
        # 和 output.json 落在同一目录；未提供时回退 task_dir 保持既有行为。
        output_dir = getattr(args, "output_dir", None) or args.task_dir
        self.interface = DataGenHypInterface(args.config, work_dir=output_dir, task_name=args.task_name)
        self.session_id = None
        self.status = None
    
    # 加载任务：用户提供数据集信息（必填）和任务描述（可选），从 prompt.json 中提取字段
    async def load_task(self):
        """Load task and create MAS session for geography discovery.

        新版逻辑：
        - 从任务目录下的 prompt.json 读取任务与数据集配置；
        - 将原先简单的 spatial_extent 改为完整的 dataset_info 结构，
          作为后续各个 Agent 之间传递的数据集元信息载体。
        """
        self.logger.info(f"Creating geography discovery session for: {self.args.task_dir}")
        
        await self.interface.startup()
        
        task_desc_path = osp.join(self.args.task_dir, "prompt.json")
        if not osp.exists(task_desc_path):
            raise FileNotFoundError(f"Task description not found: {task_desc_path}")
        
        with open(task_desc_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        
        # ------------------------------------------------------------------
        # 1. 提取基本任务信息
        # ------------------------------------------------------------------
        goal = params.get("task_description")  # 可选
        region_name = params.get("region_name", "")
        domain = params.get("domain", "geography")

        composite_anchor = params.get("anchor_asset")
        is_composite_anchor = isinstance(composite_anchor, dict) and composite_anchor.get("type") == "composite_observation"
        if composite_anchor is not None and not isinstance(composite_anchor, dict):
            raise ValueError("anchor_asset must be an object when provided")
        if is_composite_anchor:
            multi_dataset_entries = composite_anchor.get("components") or []
            if not isinstance(multi_dataset_entries, list) or len(multi_dataset_entries) < 2:
                raise ValueError("composite anchor_asset.components must contain at least two datasets")
        else:
            multi_dataset_entries = params.get("datasets") or []
            if not multi_dataset_entries:
                raw_descriptions = params.get("data_descriptions")
                if raw_descriptions is None and isinstance(params.get("data_description"), list):
                    raw_descriptions = params.get("data_description")
                if raw_descriptions is not None:
                    if not isinstance(raw_descriptions, list):
                        raise ValueError("data_descriptions must be a list when provided")
                    raw_paths = params.get("data_paths") or []
                    if raw_paths and not isinstance(raw_paths, list):
                        raise ValueError("data_paths must be a list when provided")
                    multi_dataset_entries = []
                    for index, description in enumerate(raw_descriptions):
                        if not isinstance(description, dict):
                            raise ValueError(f"data_descriptions[{index}] must be an object")
                        multi_dataset_entries.append({
                            "role": description.get("role") or "supporting_dataset",
                            "data_description": description,
                            "data_path": description.get("data_path") or (raw_paths[index] if index < len(raw_paths) else ""),
                        })
        if multi_dataset_entries and not isinstance(multi_dataset_entries, list):
            raise ValueError("datasets must be a list when provided in prompt.json")
        if not is_composite_anchor and len(multi_dataset_entries) > 1:
            component_roles = [
                entry.get("role") or "supporting_dataset"
                for entry in multi_dataset_entries
                if isinstance(entry, dict)
            ]
            composite_anchor = {
                "id": "composite_observation",
                "type": "composite_observation",
                "name": params.get("composite_name") or "Composite observation",
                "observable_group": params.get("observable_group") or "multidataset_observation",
                "data_description": {
                    "dataset_name": params.get("composite_name") or "Composite observation",
                    "data_type": "Composite multi-dataset observation",
                    "data_format": "Role-labeled, spatially and temporally aligned dataset bundle",
                    "spatial_coverage": {"region_name": region_name},
                },
                "components": multi_dataset_entries,
                "combination_contract": {
                    "joint_observable": params.get("joint_observable") or "The declared components jointly define the task observation.",
                    "required_roles": component_roles,
                    "component_count": len(multi_dataset_entries),
                },
            }
            is_composite_anchor = True
        primary_entry = multi_dataset_entries[0] if multi_dataset_entries else {}
        if primary_entry and not isinstance(primary_entry, dict):
            raise ValueError("datasets entries must be objects")

        # data_description 在新任务中可能是字符串，也可能是一个结构化 dict
        legacy_data_description = params.get("data_description")
        if isinstance(legacy_data_description, list):
            legacy_data_description = None
        raw_data_description = (
            composite_anchor.get("data_description")
            if is_composite_anchor
            else legacy_data_description or primary_entry.get("data_description")
        )
        if is_composite_anchor and not raw_data_description:
            # Composite prompts may describe each required component without
            # repeating a redundant top-level data_description block.
            combination_contract = composite_anchor.get("combination_contract")
            if not isinstance(combination_contract, dict):
                combination_contract = {}
            required_roles = combination_contract.get("required_roles")
            if not isinstance(required_roles, list):
                required_roles = [
                    entry.get("role")
                    for entry in multi_dataset_entries
                    if isinstance(entry, dict) and entry.get("role")
                ]
            raw_data_description = {
                "dataset_name": composite_anchor.get("name") or "Composite observation",
                "data_type": "Composite multi-dataset observation",
                "data_format": "Role-labeled, spatially and temporally aligned dataset bundle",
                "spatial_coverage": {"region_name": region_name},
                "required_roles": required_roles,
                "component_count": len(multi_dataset_entries),
            }
        declared_data_path = (
            ""
            if is_composite_anchor
            else str(params.get("data_path") or primary_entry.get("data_path") or "").strip()
        )
        source_prefix, separator, source_identifier = declared_data_path.partition(":")
        normalized_source_prefix = source_prefix.strip().upper() if separator else ""
        source_identifier = source_identifier.strip() if separator else ""
        gee_id_from_data_path = ""
        external_source = ""
        external_id = ""
        local_path_from_data_path = ""
        if normalized_source_prefix == "GEE":
            gee_id_from_data_path = source_identifier
            if not gee_id_from_data_path:
                raise ValueError("GEE data_path must include an Earth Engine asset id")
        elif normalized_source_prefix == "LOCAL":
            local_path_from_data_path = source_identifier
            if not local_path_from_data_path:
                raise ValueError("LOCAL data_path must include a local path")
        elif normalized_source_prefix in EXTERNAL_DATASET_SOURCES:
            external_source = normalized_source_prefix
            external_id = source_identifier
            if not external_id:
                raise ValueError(f"{external_source} data_path must include a dataset identifier")

        explicit_gee_id = str(
            params.get("gee_id")
            or params.get("ee_id")
            or params.get("gee_asset_id")
            or ""
        ).strip()
        if explicit_gee_id and gee_id_from_data_path and explicit_gee_id != gee_id_from_data_path:
            raise ValueError(
                "Conflicting Earth Engine asset ids in data_path and gee_id/ee_id: "
                f"{gee_id_from_data_path!r} != {explicit_gee_id!r}"
            )
        gee_id = explicit_gee_id or gee_id_from_data_path
        data_path = "" if gee_id_from_data_path or external_source else (local_path_from_data_path or declared_data_path)
        gee_ctor = str(
            params.get("gee_ctor")
            or params.get("ee_ctor")
            or ""
        ).strip()

        # 校验基础数据定位信息：至少提供本地路径或 GEE 资产
        if not raw_data_description:
            raise ValueError("data_description is required in prompt.json")
        if not is_composite_anchor and not data_path and not gee_id and not external_source:
            raise ValueError("Either a local data_path, a GEE asset, or a supported external source is required in prompt.json")

        # 统一整理为 base_dataset 所需字段
        if isinstance(raw_data_description, dict):
            base_name = (
                composite_anchor.get("name") if is_composite_anchor else raw_data_description.get("dataset_name")
            ) or region_name or "User Main Dataset"
            base_description = raw_data_description
            # 尝试从描述中提取一些常用信息（若不存在则留空，由下游 Agent 再推理）
            declared_data_type = str(raw_data_description.get("data_type", "")).lower()
            if "raster" in declared_data_type or "image" in declared_data_type:
                modality = "remote_sensing"
            elif "feature" in declared_data_type or "vector" in declared_data_type:
                modality = "vector"
            else:
                modality = "tabular"
            crs = raw_data_description.get("coordinate_system")
            time_range = raw_data_description.get("temporal_coverage")
            geometry = raw_data_description.get("spatial_coverage")
        elif isinstance(raw_data_description, str):
            # 字符串描述：使用默认值
            base_name = region_name or "User Main Dataset"
            base_description = {"description": raw_data_description}
            modality = "remote_sensing"  # 默认遥感
            crs = None
            time_range = None
            geometry = None
        else:
            raise ValueError(f"data_description must be dict or str, got {type(raw_data_description)}")

        # ------------------------------------------------------------------
        # 2. 构建 dataset_info 顶层结构
        #    约定参考 plan.md：
        #    - base_dataset
        #    - retrieved_datasets
        #    - available_datasets
        # ------------------------------------------------------------------
        dataset_type = (
            "composite_dataset" if is_composite_anchor
            else "gee_dataset" if gee_id
            else "external_dataset" if external_source
            else "local_dataset"
        )
        dataset_source = (
            composite_anchor.get("id") if is_composite_anchor else
            f"GEE:{gee_id}" if gee_id else
            (f"{external_source}:{external_id}" if external_source else (f"LOCAL:{data_path}" if data_path else ""))
        )
        base_dataset = {
            "name": base_name,
            "data_description": base_description,
            "data_path": data_path,          # LOCAL 路径
            "type": dataset_type,
            "source": dataset_source,
            "id": composite_anchor.get("id") if is_composite_anchor else external_id or gee_id or None,
            "dataset_source": external_source or None,
            "ee_id": gee_id or None,
            "gee_id": gee_id or None,
            "ee_ctor": gee_ctor or None,
            "modality": modality,            # "remote_sensing" / "tabular" / "vector" / "text" 等
            "crs": crs,
            "geometry": geometry,
            "time_range": time_range,
            "region_name": region_name,
            "domain": domain,
        }
        if is_composite_anchor:
            base_dataset["anchor_asset"] = composite_anchor
            base_dataset["anchor_asset_id"] = composite_anchor.get("id")
            base_dataset["anchor_asset_type"] = "composite_observation"
            base_dataset["combination_contract"] = composite_anchor.get("combination_contract") or {}
        if isinstance(raw_data_description, dict):
            source_metadata = raw_data_description.get("source_metadata") or {}
            if not isinstance(source_metadata, dict):
                raise ValueError("data_description.source_metadata must be an object")
            metadata_source = str(source_metadata.get("source") or "").strip().upper()
            metadata_id = str(source_metadata.get("id") or "").strip()
            if external_source and metadata_source and metadata_source != external_source:
                raise ValueError("data_path and source_metadata.source identify different external sources")
            if external_id and metadata_id and metadata_id != external_id:
                raise ValueError("data_path and source_metadata.id identify different external datasets")
            for metadata_key in ("doc_url", "stac_url", "provider", "license", "terms_of_use", "thematic_group", "overpass_ql", "endpoint"):
                if source_metadata.get(metadata_key) is not None:
                    base_dataset[metadata_key] = source_metadata[metadata_key]
        # canonical_id：base 身份的唯一真相源，由代码从 prompt.json 派生，
        # 供下游四套 base 识别统一按 id 判定，不受 LLM 简称/伪造身份影响。
        if not is_composite_anchor:
            base_dataset["canonical_id"] = plan_utils.derive_base_canonical_id(base_dataset)

        dataset_components = []
        for index, entry in enumerate(multi_dataset_entries):
            if not isinstance(entry, dict):
                raise ValueError("datasets entries must be objects")
            component_description = entry.get("data_description")
            component_path = str(entry.get("data_path") or "").strip()
            if not isinstance(component_description, dict):
                raise ValueError(f"datasets[{index}].data_description must be an object")
            if not component_path:
                raise ValueError(f"datasets[{index}].data_path is required")
            component_prefix, component_separator, component_identifier = component_path.partition(":")
            component_prefix = component_prefix.strip().upper() if component_separator else ""
            component_identifier = component_identifier.strip() if component_separator else ""
            component_gee_id = component_identifier if component_prefix == "GEE" else ""
            component_local_path = component_identifier if component_prefix == "LOCAL" else ""
            component_external_source = component_prefix if component_prefix in EXTERNAL_DATASET_SOURCES else ""
            component_external_id = component_identifier if component_external_source else ""
            if component_prefix in {"GEE", *EXTERNAL_DATASET_SOURCES} and not component_identifier:
                raise ValueError(f"datasets[{index}].data_path must include a dataset identifier")
            if component_prefix == "LOCAL" and not component_local_path:
                raise ValueError(f"datasets[{index}].LOCAL data_path must include a local path")
            component_data_path = component_local_path or component_path
            component_name = component_description.get("dataset_name") or f"Dataset component {index + 1}"
            component_type = str(component_description.get("data_type", "")).lower()
            component_modality = (
                "remote_sensing" if "raster" in component_type or "image" in component_type
                else "vector" if "feature" in component_type or "vector" in component_type
                else "tabular"
            )
            component = {
                "role": entry.get("role") or "supporting_dataset",
                "name": component_name,
                "data_description": component_description,
                "data_path": "" if component_gee_id or component_external_source else component_data_path,
                "type": "gee_dataset" if component_gee_id else ("external_dataset" if component_external_source else "local_dataset"),
                "source": f"GEE:{component_gee_id}" if component_gee_id else (f"{component_external_source}:{component_external_id}" if component_external_source else f"LOCAL:{component_data_path}"),
                "id": component_external_id or component_gee_id or None,
                "dataset_source": component_external_source or None,
                "ee_id": component_gee_id or None,
                "gee_id": component_gee_id or None,
                "modality": component_modality,
                "region_name": region_name,
                "domain": domain,
            }
            component["canonical_id"] = plan_utils.derive_base_canonical_id(component)
            dataset_components.append(component)
        if dataset_components:
            base_dataset["dataset_components"] = dataset_components
        if is_composite_anchor:
            component_ids = [
                str(component.get("canonical_id") or component.get("source") or component.get("name") or "").strip()
                for component in dataset_components
            ]
            component_ids = [component_id for component_id in component_ids if component_id]
            if len(component_ids) < 2:
                raise ValueError("composite anchor requires at least two canonical dataset components")
            base_dataset["canonical_id"] = "composite:" + "|".join(sorted(component_ids))

        dataset_info = {
            "base_dataset": base_dataset,
            # ResourcesCheck / GeoDataSearch 填充
            "retrieved_datasets": [],
            # 为了让下游 Agent 快速知道目前有哪些数据可用
            "available_datasets": [base_name] + [item["name"] for item in dataset_components if item["name"] != base_name],
        }
        if dataset_components:
            dataset_info["dataset_components"] = dataset_components
        if is_composite_anchor:
            dataset_info["anchor_asset"] = composite_anchor
            dataset_info["anchor_asset_id"] = composite_anchor.get("id")
            dataset_info["anchor_asset_type"] = "composite_observation"

        # ------------------------------------------------------------------
        # 3. 初始时 data_requirements 和 analysis_methods 为空，
        #    由上游 Survey / DomainKnowledge 与 TASK_PLANNING 自主探索确定
        # ------------------------------------------------------------------
        data_requirements = []
        analysis_methods = []
        
        self.session_id = await self.interface.create_session(
            task_description=goal,
            background="",  
            # 下游 Agent 会通过 session.task.spatial_extent 作为 dataset_info 使用
            spatial_extent=dataset_info,
            data_requirements=data_requirements,
            analysis_methods=analysis_methods
        )
        
        self.logger.info(f"Session created: {self.session_id}")
    
    async def generate_findings(self):
        """Run MAS and return ranked application results plus the session JSON path."""
        if self.session_id is None:
            await self.load_task()
        
        await self.interface.startup()
        
        async def status_callback(session_id, old_state, new_state):
            # Agent transition logging is handled by OrchestrationAgent
            pass
        
        while self.status != "completed":
            try:
                full_status = await self.interface.get_session_status(self.session_id)
                self.status = full_status['state']
                iterations = full_status['iterations_completed']
                
                if self.status == "awaiting_feedback":
                    if self.args.offline_feedback:
                        with open(self.args.offline_feedback, "r") as f:
                            feedback = json.load(f)
                        await self.interface.add_feedback(self.session_id, feedback)
                        self.logger.info(f"Feedback added: {feedback}")
                
                elif self.status == "completed":
                    self.logger.info("Discovery session completed")
                    break
                
                elif self.status == "error":
                    raise RuntimeError("Error in MAS session")
                
                self.logger.info(f"Running session {self.session_id}, iteration {iterations}, state: {self.status}")
                self.status = await self.interface.run_session(
                    self.session_id,
                    status_callback=status_callback
                )
                
            except Exception as e:
                self.logger.error(f"Error in session: {str(e)}")
                raise
        
        top_findings = await self.interface.get_top_findings(self.session_id)
        self.logger.info(f"Generated {len(top_findings)} ranked applications")
        
        # Save findings
        # Session files are saved in results/{task_name}/traj_{self.session_id}/ by the memory manager
        session_json = osp.join("results", self.args.task_name, f"traj_{self.session_id}", f"traj_{self.session_id}.json")
        
        return top_findings, session_json
