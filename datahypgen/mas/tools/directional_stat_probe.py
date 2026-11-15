"""方向性主张的微型实证探针：把"组间对比"类质询 finding 映射为一条可计算的统计量。

用途与边界：
- 只服务单条 finding 的当轮修订：结果挂在 finding 的 ``micro_empirical_check`` 字段，
  随 findings_json 进入该轮修订 prompt，不写入 evidence_context / 树 / 全局配置。
- 泛化设计：探针本身不知道任何研究领域。一切通过列名驱动——spec 抽取调用只能从
  数据集画像给出的列清单里选列，计算端是纯函数（分组中位数/均值 + Mann-Whitney U），
  因此对任何列式数据上的"组 A 相对组 B 的结果变量方向"问题同样适用。
- 可插拔：注册进 tool_registry；模型调用以 ``model_call`` 可调用注入，不绑定具体 agent。

失败语义：抽取不可计算（非对比类主张、列不存在）返回 None；任何异常由调用方吞掉
（探针失败不得中断成熟化），只记 warning。
"""

import json
import logging
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .tool_registry import register_tool

logger = logging.getLogger("DirectionalStatProbe")

# spec 抽取与摘要的单次输出预算：spec 是六七个短字段的 JSON，200 tokens 只在模型
# 啰嗦时才可能吃紧；摘要复用 spec 调用的观测结果，不再二次调用。
_SPEC_MAX_TOKENS = 300

_SIGN_GT = "a_gt_b"
_SIGN_LT = "a_lt_b"
_SIGN_NONE = "no_difference"


def build_directional_directive(spec: Dict[str, Any], contrast: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从 spec+contrast 确定性导出修订必须遵守的方向约束（纯函数，不经 LLM）。

    这是"生成端正确性"的载体：修订器动笔前读到的不是一段原始统计，而是一条可执行
    指令——存续主张的允许符号、可直接采纳的修正句式（observed_fragment）、以及退路
    （对该对照撤销方向性断言）。任何方向的取舍都由代码给出，模型只负责措辞。
    """
    if contrast.get("status") != "ok":
        return None
    outcome = str(contrast.get("outcome_column") or spec.get("outcome_column") or "")
    a_key = str(contrast.get("group_a") or spec.get("group_a") or "")
    b_key = str(contrast.get("group_b") or spec.get("group_b") or "")
    on_filter = f" on {contrast['row_filter']}" if contrast.get("row_filter") else ""
    comparison = f"{outcome}: {a_key} vs {b_key}{on_filter}"
    observed = contrast.get("observed_sign")
    p = contrast.get("mwu_p_two_sided")
    p_text = f", MWU p={p:.3g}" if isinstance(p, (int, float)) else ""
    if observed in (_SIGN_GT, _SIGN_LT):
        word = "HIGHER" if observed == _SIGN_GT else "LOWER"
        # 中位数并列（二元/稀有结果）时补充均值/率，否则 fragment 只有 0 vs 0 没有信息量
        tie = contrast.get("median_a") == contrast.get("median_b")
        stat_text = f"median {contrast.get('median_a')} vs {contrast.get('median_b')}"
        if tie and contrast.get("mean_a") is not None:
            stat_text += f", mean {contrast.get('mean_a'):.3g} vs {contrast.get('mean_b'):.3g}"
        stat_text += p_text
        return {
            "comparison": comparison,
            "observed_sign": observed,
            "claimed_sign": spec.get("claimed_sign"),
            "sign_agreement": observed == spec.get("claimed_sign"),
            "required_rule": (
                f"any surviving directional claim about this comparison must state {observed} "
                f"({a_key} shows {word} {outcome} than {b_key}); to keep a different direction, "
                "withdraw the directional assertion for this comparison instead of restating it"
            ),
            "observed_fragment": (
                f"{a_key} shows {word} {outcome} than {b_key}{on_filter} ({stat_text})"
            ),
            # 修复文本必须逐字包含的核心比较子句（终检 fragment 绑定的锚点）；
            # 保语义换皮（主语互换）无法同时包含这个子句，从而文本与承诺不再可能分叉。
            "fragment_core": f"{a_key} shows {word} {outcome} than {b_key}",
        }
    if observed == _SIGN_NONE:
        return {
            "comparison": comparison,
            "observed_sign": observed,
            "claimed_sign": spec.get("claimed_sign"),
            "sign_agreement": spec.get("claimed_sign") == _SIGN_NONE,
            "required_rule": (
                f"the computed statistic finds no significant difference{p_text}; a directional "
                "claim about this comparison must be withdrawn (surviving_sign=withdrawn) or "
                "restated without the ordering"
            ),
            "observed_fragment": (
                f"no significant difference in {outcome} between {a_key} and {b_key}{p_text}"
            ),
        }
    return None


@register_tool("DirectionalStatProbe")
class DirectionalStatProbe:
    """列式数据上的组间对比探针。

    Args:
        model_call: 具备 schema 参数的异步模型调用（与 BaseAgent._call_model 同签名
            子集），用于把 finding 文本映射为对比 spec；传 None 则探针只可用
            compute_contrast 纯函数路径。
        max_rows: 读取本地表的上限，防止大文件拖垮成熟化轮。
    """

    def __init__(
        self,
        model_call: Optional[Callable[..., Awaitable[Dict[str, Any]]]] = None,
        max_rows: int = 20000,
    ):
        self.model_call = model_call
        self.max_rows = max_rows

    # ------------------------------------------------------------------
    # 列画像（确定性）：数值列给 min/median/max，低基数文本列给 top 取值。
    # 让 spec 抽取把分组 bin 选在数据实际范围内（避免 B 侧空组）。
    # ------------------------------------------------------------------
    @staticmethod
    def column_profile(df: pd.DataFrame, columns: List[str], max_values: int = 6) -> Dict[str, Any]:
        profile: Dict[str, Any] = {}
        n_rows = max(len(df), 1)
        for col in columns:
            if col not in df.columns:
                continue
            s = df[col]
            numeric = pd.to_numeric(s, errors="coerce").dropna()
            if numeric.size >= 0.5 * n_rows:
                profile[col] = {
                    "type": "numeric",
                    "min": round(float(numeric.min()), 3),
                    "p50": round(float(numeric.median()), 3),
                    "max": round(float(numeric.max()), 3),
                }
            else:
                top = s.astype(str).value_counts().head(max_values)
                profile[col] = {"type": "categorical", "top_values": [str(v) for v in top.index]}
        return profile

    # ------------------------------------------------------------------
    # spec 抽取（唯一用到 LLM 的一步；列名被约束在画像列清单内）
    # ------------------------------------------------------------------
    async def extract_contrast_spec(
        self,
        *,
        question: str,
        failure_hypothesis: str,
        directional_statement: str,
        columns: List[str],
        column_profile: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """把一条挑战方向性主张的 finding 映射为对比 spec；不可计算返回 None。"""
        if self.model_call is None or not columns:
            return None
        column_enum = {"type": "string", "enum": list(columns)}
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "extractable": {"type": "boolean"},
                # 无分组列时（单变量方向性主张）group_column 允许 "__none__"
                "outcome_column": column_enum,
                "group_column": {"type": "string", "enum": [*columns, "__none__"]},
                "group_a": {"type": "string", "maxLength": 60},
                "group_b": {"type": "string", "maxLength": 60},
                # 主张隐含的子总体（可选，一元等值过滤）：如"触发泥石流的事件"
                # -> filter_column=Response, filter_value=1。不过滤会测错总体。
                "filter_column": {"type": "string", "enum": [*columns, "__none__"]},
                "filter_value": {"type": "string", "maxLength": 60},
                # 主张文本中断言方向的原词组（如 "lower rainfall intensities"），
                # 供验收门做新文案的方向残留检查。
                "claimed_phrase": {"type": "string", "maxLength": 160},
                "claimed_sign": {"type": "string", "enum": [_SIGN_GT, _SIGN_LT, _SIGN_NONE]},
            },
            "required": [
                "extractable", "outcome_column", "group_column",
                "group_a", "group_b", "filter_column", "filter_value",
                "claimed_phrase", "claimed_sign",
            ],
        }
        profile_section = (
            "Column value ranges/levels (choose group bins and filter values INSIDE these):\n"
            f"{json.dumps(column_profile, ensure_ascii=False)}\n\n" if column_profile else ""
        )
        prompt = (
            "Map one research-claim challenge to a two-group contrast on a tabular dataset, "
            "or mark it not extractable.\n\n"
            f"Challenged directional claim: {directional_statement}\n"
            f"Challenge question: {question}\n"
            f"Failure hypothesis: {failure_hypothesis}\n\n"
            f"Available columns: {columns}\n\n"
            f"{profile_section}"
            "Rules: outcome_column is the numeric variable the claim asserts something about. "
            "group_column is the column that splits rows into the two sides the claim compares "
            "(a year indicator, a class label, a threshold split...); use __none__ only when the "
            "claim compares a single distribution to a fixed reference. group_a / group_b are the "
            "concrete values of that column on each side, phrased exactly as they appear in the "
            "data. A group value may carry a comparator prefix (>=, <=, !=, >, <) for an "
            "open-ended side, e.g. group_b='!=2019' for 'all other years' or group_b='>=1' for "
            "'second and later years'. "
            "A claim that a threshold, requirement, or sensitivity rises/falls with some condition "
            "is extractable: outcome_column is the measured quantity whose change constitutes that "
            "threshold (e.g. the triggering intensity), group_column is the variable encoding the "
            "condition (e.g. time-since-fire), and group_a/group_b are two condition bins compared "
            "with comparator prefixes (e.g. '<90' vs '>=365'). 'Decreases with X' means the "
            "early/low-X side shows HIGHER outcome (a_gt_b with a=early bin) unless the claim says "
            "otherwise. "
            "outcome_column must be the measured quantity the claim argues about — NEVER the "
            "variable that encodes the condition itself (a claim about outcomes differing by "
            "time/cohorts compares the OUTCOME between groups, not the time values). A claim "
            "about a slope/decay/rate-of-change difference between cohorts is extractable only as "
            "a one-comparison proxy that decides it (e.g. compare the outcome between cohorts "
            "WITHIN the late/high-condition regime via filter_column); if no single comparison "
            "decides it, set extractable=false. "
            "filter_column / filter_value fix the comparison population; decide it from the "
            "claim's subject with this procedure: if the claim asserts a property OF the outcome "
            "events themselves (e.g. 'debris-flow events will be triggered by lower intensities' "
            "— the triggering intensities OF debris-flow occurrences), the population is the rows "
            "where that outcome occurred, so set filter_column to the outcome-label column and "
            "filter_value to its positive value (e.g. Response=1). Use __none__ only when the "
            "sign is asserted about a quantity measured on every row (e.g. 'basins with higher "
            "mdNBR produce more events' compares the binary outcome across all rows, so outcome "
            "column is the label itself and no filter applies). Getting the population wrong "
            "reverses the statistic — when the claim's subject names the outcome, filtering is "
            "mandatory. "
            "claimed_sign states whether the claim says group_a's outcome is greater "
            "(a_gt_b) or smaller (a_lt_b) than group_b's, or asserts no ordering (no_difference). "
            "claimed_phrase quotes the comparative wording from the claim. Set extractable=false "
            "when no column pair can decide the claim (e.g. it needs an unlisted external variable "
            "or a spatial/raster quantity)."
        )
        try:
            result = await self.model_call(
                prompt=prompt,
                system_prompt=(
                    "You translate a challenged scientific claim into one minimal group-contrast "
                    "specification over the listed columns. You do not judge the claim."
                ),
                schema=schema,
                temperature=0.0,
                max_tokens=_SPEC_MAX_TOKENS,
            )
        except Exception as exc:
            logger.warning("contrast spec extraction failed: %s", exc)
            return None
        if not isinstance(result, dict) or not result.get("extractable"):
            return None
        if result.get("outcome_column") not in columns:
            return None
        # 结果变量守卫：outcome 不得与分组列或过滤列相同——把条件变量当结果变量
        # （如火龄当 outcome 比较队列）会产出"确认无意义对比"的假统计。
        if result.get("group_column") not in ("__none__", None) and result.get("outcome_column") == result.get("group_column"):
            return None
        if result.get("filter_column") not in ("__none__", None) and result.get("outcome_column") == result.get("filter_column"):
            return None
        group_column = result.get("group_column")
        if group_column not in columns:
            result["group_column"] = "__none__"
        return result

    # ------------------------------------------------------------------
    # 确定性计算（纯函数，无 LLM、无 IO）
    # ------------------------------------------------------------------
    @staticmethod
    def compute_contrast(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
        """按 spec 计算一行组间统计。列缺失/样本不足时返回 status=unavailable。"""
        outcome = str(spec.get("outcome_column") or "")
        if outcome not in df.columns:
            return {"status": "unavailable", "reason": f"outcome column missing: {outcome}"}
        col = pd.to_numeric(df[outcome], errors="coerce").dropna()
        group_column = str(spec.get("group_column") or "__none__")
        filter_column = str(spec.get("filter_column") or "__none__")
        filter_value = str(spec.get("filter_value") or "").strip()
        if filter_column != "__none__" and filter_column in df.columns and filter_value:
            g_all = df[filter_column]
            numeric_all = pd.to_numeric(g_all, errors="coerce")
            direct = g_all.astype(str).str.strip() == filter_value
            try:
                mask = direct | (numeric_all == float(filter_value))
            except ValueError:
                mask = direct
            df = df.loc[mask]
        if group_column == "__none__" or group_column not in df.columns:
            col = pd.to_numeric(df[outcome], errors="coerce").dropna()
            return {
                "status": "ok",
                "outcome_column": outcome,
                "group_column": None,
                "row_filter": None if filter_column == "__none__" else f"{filter_column}=={filter_value}",
                "n": int(col.size),
                "median": float(col.median()) if col.size else None,
                "p25": float(col.quantile(0.25)) if col.size else None,
                "p75": float(col.quantile(0.75)) if col.size else None,
                "observed_sign": None,
            }
        a_key, b_key = str(spec.get("group_a") or ""), str(spec.get("group_b") or "")
        g = df[group_column]

        def _mask(side: str) -> pd.Series:
            """组侧取值匹配：支持比较符前缀（>=/<=/!=/>/<，面向"其余年份"这类开放侧）
            与裸值（字符串等值或数值等值，兼容数值列的文本写法）。"""
            normalized = str(side).strip()
            numeric = pd.to_numeric(g, errors="coerce")
            for op in (">=", "<=", "!=", ">", "<"):
                if normalized.startswith(op):
                    threshold = float(normalized[len(op):].strip())
                    return {
                        ">=": numeric >= threshold,
                        "<=": numeric <= threshold,
                        "!=": numeric != threshold,
                        ">": numeric > threshold,
                        "<": numeric < threshold,
                    }[op]
            direct = g.astype(str).str.strip() == normalized
            try:
                return direct | (numeric == float(normalized))
            except ValueError:
                return direct

        a = pd.to_numeric(df.loc[_mask(a_key), outcome], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[_mask(b_key), outcome], errors="coerce").dropna()
        bin_repair = None
        if (a.size < 3 or b.size < 3) and group_column != "__none__":
            # 确定性 bin 修复：主张文本里的时间框架常超出数据实际范围（如 ">365" 而数据
            # 最大 282）。同一分组列、同一过滤总体内改用中位数分割，语义保持"低条件侧 vs
            # 高条件侧"；修复方式记录在结果里供审计。
            g_num = pd.to_numeric(df[group_column], errors="coerce")
            if g_num.notna().sum() >= 6:
                cut = float(g_num.dropna().median())
                a = pd.to_numeric(df.loc[g_num <= cut, outcome], errors="coerce").dropna()
                b = pd.to_numeric(df.loc[g_num > cut, outcome], errors="coerce").dropna()
                a_key, b_key = f"<={cut:g}", f">{cut:g}"
                bin_repair = f"median_split@{cut:g} (spec bins were {spec.get('group_a')} / {spec.get('group_b')})"
        result: Dict[str, Any] = {
            "status": "ok",
            "outcome_column": outcome,
            "group_column": group_column,
            "row_filter": None if filter_column == "__none__" else f"{filter_column}=={filter_value}",
            "group_a": a_key,
            "n_a": int(a.size),
            "median_a": float(a.median()) if a.size else None,
            "group_b": b_key,
            "n_b": int(b.size),
            "median_b": float(b.median()) if b.size else None,
        }
        if bin_repair:
            result["bin_repair"] = bin_repair
        if a.size < 3 or b.size < 3:
            result.update(status="unavailable", reason="group sample < 3")
            return result
        _, p_two_sided = stats.mannwhitneyu(a, b, alternative="two-sided")
        result["mean_a"] = float(a.mean())
        result["mean_b"] = float(b.mean())
        result["mwu_p_two_sided"] = float(p_two_sided)
        if p_two_sided >= 0.05:
            result["observed_sign"] = _SIGN_NONE
        else:
            # 符号判定：中位数优先（重尾稳健）；中位数并列（二元/稀有结果两侧都是 0）
            # 时用均值回退——二元率对比的中位数全为 0，按中位数会给错方向。
            if float(a.median()) != float(b.median()):
                result["observed_sign"] = _SIGN_GT if float(a.median()) > float(b.median()) else _SIGN_LT
            elif a.mean() != b.mean():
                result["observed_sign"] = _SIGN_GT if a.mean() > b.mean() else _SIGN_LT
            else:
                result["observed_sign"] = _SIGN_NONE
        return result

    # ------------------------------------------------------------------
    # 本地表读取（与 dataset_profiler 同源：base_dataset.data_path）
    # ------------------------------------------------------------------
    def load_dataframe(self, data_path: str) -> Optional[pd.DataFrame]:
        if not data_path:
            return None
        path = data_path
        if os.path.isdir(path):
            candidates = sorted(
                f for f in os.listdir(path)
                if f.lower().endswith((".csv", ".xlsx", ".xls", ".parquet"))
            )
            if not candidates:
                return None

            # 表选择启发：主事件/观测表优先（名字含 event/main/record/data），
            # 特征枚举/组合/画像表降权，同分时 csv 优先（xlsx 常是多副本工作簿）。
            def _rank(name: str) -> int:
                base = name.lower().rsplit(".", 1)[0]
                score = 0
                if any(key in base for key in ("event", "main", "record")) or base == "data":
                    score -= 20
                if any(key in base for key in ("feature", "combination", "profile", "portrait", "sample")):
                    score += 20
                if not name.lower().endswith(".csv"):
                    score += 5
                return score

            path = os.path.join(path, min(candidates, key=_rank))
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            elif path.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
            elif path.lower().endswith(".parquet"):
                df = pd.read_parquet(path)
            else:
                return None
        except Exception as exc:
            logger.warning("probe failed to load %s: %s", path, exc)
            return None
        if len(df) > self.max_rows:
            df = df.iloc[: self.max_rows]
        return df

    # ------------------------------------------------------------------
    # 一步式入口：finding -> 一行统计（attach 到 finding 的内容）
    # ------------------------------------------------------------------
    async def probe_finding(
        self,
        *,
        df: pd.DataFrame,
        question: str,
        failure_hypothesis: str,
        directional_statement: str,
        columns: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        spec = await self.extract_contrast_spec(
            question=question,
            failure_hypothesis=failure_hypothesis,
            directional_statement=directional_statement,
            columns=columns if columns is not None else list(df.columns),
            column_profile=self.column_profile(df, columns if columns is not None else list(df.columns)),
        )
        if spec is None:
            return None
        contrast = self.compute_contrast(df, spec)
        return {
            "spec": spec,
            "contrast": contrast,
            # 生成端约束：由 spec+contrast 确定性导出，不经 LLM。修订器在动笔前读到
            # 这条指令，而不是事后被验收门拦截。
            "directive": build_directional_directive(spec, contrast),
        }
