"""
领域知识智能体

该智能体接收文献检索智能体的输出（逐篇论文列表 + 聚合视图），
根据现有的相关论文进行深入分析，生成领域知识库，包括研究背景综述、已覆盖的簇、发现的空白和建议，为后续新的研究问题生成奠定知识背景。
"""

import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class DomainKnowledgeAgent(BaseAgent):
    """
    领域知识智能体：将文献内容整合为领域知识库。
    
    功能：
    - 分析文献检索结果
    - 生成研究背景综述
    - 识别方法、应用、数据集簇
    - 发现自然因素组合研究空白
    - 提出后续研究建议
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the domain knowledge agent.
        
        Args:
            model: Language model to use
            config: Configuration dictionary
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.max_analysis_papers = config.get("max_analysis_papers", 20)
        self.min_filtered_papers = config.get("min_filtered_papers", 3)
        self.output_format = config.get("output_format", "json")
        self.analysis_max_tokens = int(config.get("analysis_max_tokens", 3072))

    def _sort_papers_by_score(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(papers, key=lambda x: x.get("score", 0), reverse=True)

    def _select_analysis_papers(
        self,
        survey_results: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, int | str]]:
        filtered_papers = survey_results.get("papers", []) or []
        raw_papers = survey_results.get("papers_all", []) or []

        sorted_filtered = self._sort_papers_by_score(filtered_papers)
        if len(sorted_filtered) >= self.min_filtered_papers or len(raw_papers) <= len(sorted_filtered):
            selected_papers = sorted_filtered[:self.max_analysis_papers]
            return selected_papers, {
                "filtered_count": len(filtered_papers),
                "raw_count": len(raw_papers),
                "selected_count": len(selected_papers),
                "raw_backfill_count": 0,
                "selection_mode": "filtered_only",
            }

        sorted_raw = self._sort_papers_by_score(raw_papers)
        selected_papers: List[Dict[str, Any]] = []
        seen_ids = set()

        for paper in sorted_filtered:
            paper_id = paper.get("id")
            if paper_id in seen_ids:
                continue
            selected_papers.append(paper)
            seen_ids.add(paper_id)
            if len(selected_papers) >= self.max_analysis_papers:
                break

        raw_backfill_count = 0
        for paper in sorted_raw:
            if len(selected_papers) >= self.max_analysis_papers:
                break
            paper_id = paper.get("id")
            if paper_id in seen_ids:
                continue
            selected_papers.append(paper)
            seen_ids.add(paper_id)
            raw_backfill_count += 1

        return selected_papers, {
            "filtered_count": len(filtered_papers),
            "raw_count": len(raw_papers),
            "selected_count": len(selected_papers),
            "raw_backfill_count": raw_backfill_count,
            "selection_mode": "filtered_plus_raw_backfill",
        }
    
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行领域知识分析。
        
        Args:
            context: 包含 survey_agent 输出
            params: 额外参数
            
        Returns:
            结构化领域知识分析结果
        """
        survey_results = context.get("survey_results", {})
        aggregated_views = survey_results.get("aggregated_views", {})
        top_papers, selection_stats = self._select_analysis_papers(survey_results)

        logger.info(
            "DomainKnowledgeAgent: filtered=%s, raw=%s, selected=%s, raw_backfill=%s, "
            "mode=%s, max_analysis=%s",
            selection_stats["filtered_count"],
            selection_stats["raw_count"],
            selection_stats["selected_count"],
            selection_stats["raw_backfill_count"],
            selection_stats["selection_mode"],
            self.max_analysis_papers,
        )
        
        # 生成领域知识分析
        analysis = await self._analyze_domain_knowledge(top_papers, aggregated_views, context)

        # 将 triples 从 aggregated_views 传递到输出中，供 Plan Agent 使用
        triples = aggregated_views.get("method_application_dataset_triples", [])
        if triples:
            analysis["method_application_dataset_triples"] = triples
            logger.info(f"DomainKnowledgeAgent: Passing {len(triples)} method-application-dataset triples to downstream agents")

        logger.info(f"DomainKnowledgeAgent: Generated domain knowledge analysis with {len(analysis.get('clusters', {}).get('applications', {}))} application clusters")

        return analysis
    
    async def _analyze_domain_knowledge(self, 
                                      papers: List[Dict[str, Any]], 
                                      aggregated_views: Dict[str, Any],
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析领域知识，生成结构化输出。
        
        Args:
            papers: Top papers list
            aggregated_views: Aggregated clusters
            context: Original context
            
        Returns:
            Structured analysis
        """
        # 构建输入数据
        input_data = {
            "papers": papers,
            "aggregated_views": aggregated_views,
            "domain": context.get("domain", ""),
            "task_description": context.get("description", ""),
            "data_description": context.get("data_description", "")
        }
        
        # 构建 prompt
        prompt = self._build_analysis_prompt(input_data)
        
        # 定义输出 schema
        output_schema = {
            "type": "object",
            "properties": {
                "background_summary": {
                    "type": "string",
                    "description": "研究背景综述",
                    "maxLength": 1200,
                },
                "clusters": {
                    "type": "object",
                    "properties": {
                        "methods": {
                            "type": "object",
                            "description": "方法簇：methodological_framework -> [paper_ids]。键名应描述分析范式或推理策略（如'空间自相关感知的集成学习'、'基于反事实的因果推断'），而非软件库名称或数据集名称。",
                            "additionalProperties": {
                                "type": "array",
                                "items": {"type": "string", "maxLength": 80},
                                "maxItems": 12,
                            },
                        },
                        "applications": {
                            "type": "object",
                            "description": "应用场景簇：application -> [paper_ids]",
                            "additionalProperties": {
                                "type": "array",
                                "items": {"type": "string", "maxLength": 80},
                                "maxItems": 12,
                            },
                        },
                        "datasets": {
                            "type": "object",
                            "description": "数据集簇：dataset -> [paper_ids]",
                            "additionalProperties": {
                                "type": "array",
                                "items": {"type": "string", "maxLength": 80},
                                "maxItems": 12,
                            },
                        }
                    },
                    "required": ["methods", "applications", "datasets"],
                    "additionalProperties": False,
                },
                "gaps": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Gap identifier, e.g. gap_1", "maxLength": 40},
                            "description": {"type": "string", "maxLength": 280},
                            "missing_combination": {"type": "string", "maxLength": 180},
                            "potential_impact": {"type": "string", "maxLength": 220},
                            "cited_papers": {"type": "array", "items": {"type": "string", "maxLength": 80}, "maxItems": 8}
                        },
                        "required": ["id"],
                        "additionalProperties": False,
                    },
                    "description": "发现的空白/缺口列表"
                },
                "suggestions": {
                    "type": "array",
                    "maxItems": 10,
                    "items": {
                        "type": "object",
                        "properties": {
                            "gap_id": {"type": "string", "description": "The gap.id this suggestion operationalizes", "maxLength": 40},
                            "research_question": {"type": "string", "maxLength": 240},
                            "hypothesis": {"type": "string", "maxLength": 220},
                            "proposed_method": {"type": "string", "maxLength": 220},
                            "data_requirements": {"type": "string", "maxLength": 220},
                            "rationale": {"type": "string", "maxLength": 220}
                        },
                        "required": ["gap_id"],
                        "additionalProperties": False,
                    },
                    "description": "后续研究建议"
                }
            },
            "required": ["background_summary", "clusters", "gaps", "suggestions"],
            "additionalProperties": False,
        }
        
        try:
            response = await self._call_model(
                prompt=prompt,
                schema=output_schema,
                temperature=0.2,
                max_tokens=self.analysis_max_tokens,
            )
            response = self._normalize_gap_suggestion_links(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in domain knowledge analysis: {str(e)}")
            raise AgentExecutionError(f"Failed to analyze domain knowledge: {str(e)}")
    
    def _normalize_gap_suggestion_links(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(response, dict):
            return response

        gaps = response.get("gaps", [])
        suggestions = response.get("suggestions", [])
        if not isinstance(gaps, list):
            gaps = []
        if not isinstance(suggestions, list):
            suggestions = []

        normalized_gaps: List[Dict[str, Any]] = []
        gap_ids: List[str] = []
        for gap_index, gap in enumerate(gaps):
            if not isinstance(gap, dict):
                continue
            normalized_gap = dict(gap)
            gap_id = str(normalized_gap.get("id") or "").strip() or f"gap_{gap_index + 1}"
            normalized_gap["id"] = gap_id
            normalized_gaps.append(normalized_gap)
            gap_ids.append(gap_id)

        normalized_suggestions: List[Dict[str, Any]] = []
        for suggestion_index, suggestion in enumerate(suggestions):
            if not isinstance(suggestion, dict):
                continue
            normalized_suggestion = dict(suggestion)
            gap_id = str(normalized_suggestion.get("gap_id") or "").strip()
            if not gap_id or gap_id not in gap_ids:
                logger.warning(
                    "DomainKnowledgeAgent: dropping suggestion without valid explicit gap_id at index %s",
                    suggestion_index,
                )
                continue
            normalized_suggestion["gap_id"] = gap_id
            normalized_suggestions.append(normalized_suggestion)

        response["gaps"] = normalized_gaps
        response["suggestions"] = normalized_suggestions
        return response
    
    def _build_analysis_prompt(self, input_data: Dict[str, Any]) -> str:
        """
        构建分析 prompt。
        
        Args:
            input_data: 输入数据
            
        Returns:
            分析 prompt
        """
        papers = input_data["papers"]
        aggregated_views = input_data["aggregated_views"]
        domain = input_data["domain"]
        task_description = input_data["task_description"]
        data_description = input_data["data_description"]
        
        prompt = f"""
You are a senior geospatial science expert responsible for analyzing literature retrieval results and building a domain knowledge base.

**Research Domain**: {domain}
**Task Description**: {task_description}
**Dataset Description**: {data_description}

**Input Data**:
1. **Paper List** (top {len(papers)} papers, sorted by relevance):
"""
        
        for i, paper in enumerate(papers, 1):
            prompt += f"""
Paper {i} (ID: {paper['id']}, Score: {paper.get('score', 'N/A')}):
Title: {paper.get('title', 'N/A')}
Abstract: {paper.get('abstract', 'N/A')[:300]}...
Background: {paper.get('background', 'N/A')}
Contributions: {paper.get('contributions', 'N/A')}
Methods: {paper.get('methods', 'N/A')} (this should describe methodological frameworks or analytical paradigms, not library names)
Applications: {paper.get('applications', 'N/A')}
Deep-read flag: {paper.get('is_deep_read', False)}
"""
        
        prompt += f"""

2. **Aggregated Views**:
- Method clusters: {aggregated_views.get('method_clusters', {})}
- Application clusters: {aggregated_views.get('application_clusters', {})}
- Dataset clusters: {aggregated_views.get('dataset_clusters', {})}
- Example method-application-dataset triples: {aggregated_views.get('method_application_dataset_triples', [])}

**Task Requirements**:
Based on the inputs above, generate a structured domain knowledge analysis that can inspire geospatial hazard discovery. Follow all requirements below:

1. **Background summary**: Summarize the current research status, key challenges, and major trends in this domain.

2. **Covered clusters**:
   - Method clusters: identify the main methodological families and representative papers. **Method cluster keys must describe methodological frameworks or analytical paradigms** (for example, "time-series breakpoint detection and anomaly attribution", "spatial autoregressive modeling"), not software library names (pandas/sklearn) or dataset names (NDVI/ERA5).
   - Application clusters: identify the main research-problem types and representative papers.
   - Dataset clusters: identify the main dataset/data-modality families and representative papers.

3. **Discovered gaps**:
   - Analyze missing combinations across method x application x dataset.
   - Identify gaps in spatial resolution, coverage region, time period, and related observability dimensions.
   - Each gap must cite supporting paper IDs.

4. **Follow-up suggestions**:
   - Propose specific research questions and hypotheses.
   - Suggest plausible data acquisition paths and methodological chains.
   - Prefer suggestions that directly address the discovered gaps.
   - Each suggestion must explicitly point to one gap via `gap_id`.

5. **Explicit gap-suggestion linkage**:
   - Every gap must contain a unique `id` field, preferably formatted as `gap_1`, `gap_2`, ...
   - Every suggestion must contain `gap_id`, and its value must exactly reference one defined `gaps[*].id`.
   - Do not output any suggestion that is not linked to a gap.

6. **Language contract**:
   - All free-text values in the JSON output must be written in English.
   - Keep schema keys, paper IDs, and enum-like identifiers exactly as requested.
   - Keep every free-text field concise and non-redundant.
   - `background_summary`: at most 6 sentences.
   - Each gap field and suggestion field: at most 1-2 sentences per string field.

**Output format**: Follow the provided JSON schema exactly. Do not add any extra text.
When citing papers, use paper_id values such as "paper_0" or "paper_1" to reduce hallucination.
"""
        
        return prompt
