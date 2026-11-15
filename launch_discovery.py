"""
Launch DataGenHyp 
"""
import os
import os.path as osp
import sys
import json
import argparse
import asyncio
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
# import torch  # 不再需要，因为不再执行实验
import yaml
from datetime import datetime
from dotenv import load_dotenv

# Import MAS components
from datahypgen.stage import FindingGenerator
from datahypgen.mas.interface import DataGenHypInterface

load_dotenv()

# ============================================================================
# Logging Configuration
# ============================================================================
def setup_logging():
    """Setup logging configuration"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = osp.join(log_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_datahypgen.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    # pdfminer 在深读文献时对每个 PDF 页面打印 render_contents 与 XObject 明细，
    # INFO 级别下会淹没工作流日志（实测占比可达 87%）；httpcore/urllib3 同理。
    for _noisy in ("pdfminer", "httpx", "httpcore", "urllib3"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    return logging.getLogger("DataGenHyp")


# ============================================================================
# Argument Parser
# ============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="DataGenHyp Geography Discovery Pipeline: Pattern Discovery from Multi-angle Geography Data"
    )
    
    # ========================================
    # Task Configuration
    # ========================================
    task_group = parser.add_argument_group('Task Configuration')
    selection = task_group.add_mutually_exclusive_group()
    selection.add_argument(
        "--task",
        type=str,
        default="tasks/climate_surface_interaction/Planetary_Computer_Daymet_Annual_North_America",
        help="Task name or path to task directory. If it's a name, will use tasks/{task}; if it's a path, will use it directly"
    )
    selection.add_argument(
        "--benchmark", metavar="TASK_OR_GROUP",
        help="Generate a canonical benchmark task, a task group, or all tasks ('all')",
    )
    task_group.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Results output directory (defaults to results/{task_name})"
    )
    task_group.add_argument(
        "--config",
        type=str,
        default='config/debug_config.yaml',
        help="Path to configuration file"
    )
    
    # ========================================
    # Finding Generation Phase
    # ========================================
    finding_group = parser.add_argument_group('Finding Generation Phase')
    finding_group.add_argument(
        "--resume_session",
        type=str,
        default=None,
        help="If set, resume an existing MAS session instead of creating a new one (skips new finding generation)."
    )
    finding_group.add_argument(
        "--offline_feedback",
        type=str,
        default='config/feedback_global.json',
        help="Offline feedback file for finding generation"
    )

    
    return parser.parse_args()


# ============================================================================
# Main Pipeline
# ============================================================================
async def generate_benchmark_tasks(args, benchmark_root, selector, logger):
    from benchmark.core.admission import eligible_applications
    from benchmark.core.benchmark_layout import GROUPS, canonical_task_ids, grouped_task_dir, require_canonical_task_id, system_outputs_root
    from datahypgen.generation_runner import ExperimentBoundary, SystemSpec, run_system

    if args.resume_session:
        raise ValueError("Benchmark generation resumes by skipping admitted outputs. Omit --resume_session.")
    if selector == "all" or selector in GROUPS:
        task_ids = canonical_task_ids(benchmark_root)
        if selector in GROUPS:
            task_ids = [task_id for task_id in task_ids if task_id.startswith(selector + "/")]
    else:
        task_ids = [require_canonical_task_id(benchmark_root, selector)]

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if config.get("phases", {}).get("enable_plan_refinement_phase") is not True:
        raise ValueError("Benchmark generation requires enable_plan_refinement_phase")
    output_root = Path(args.output_dir).resolve() if args.output_dir else system_outputs_root(benchmark_root, "ours")
    failed = []
    for task_id in task_ids:
        output_dir = grouped_task_dir(output_root, task_id)
        target = output_dir / "full_output.json"
        try:
            if target.exists():
                eligible_applications(json.loads(target.read_text(encoding="utf-8")))
                logger.info("Skipping existing admitted output: %s", target)
                continue
            task_dir = grouped_task_dir(benchmark_root / "dataset/task_inputs", task_id)
            spec = SystemSpec(name="internagent_benchmark", family="internagent", matrix="benchmark",
                              boundary=ExperimentBoundary.REFINEMENT_TERMINAL, output_dir=output_dir, task_dir=task_dir)
            result = await run_system(spec, "benchmark_" + task_id.replace("/", "_"), args.config)
            eligible_applications(result)
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir, delete=False) as tmp:
                json.dump(result, tmp, ensure_ascii=False, indent=2)
            Path(tmp.name).replace(target)
        except Exception:
            logger.exception("Benchmark task failed: %s", task_id)
            failed.append(task_id)
    if failed:
        raise RuntimeError("Failed benchmark tasks: " + ", ".join(failed))


def main():
    logger = setup_logging()
    args = parse_arguments()
    benchmark_root = Path(__file__).resolve().parent / "benchmark"
    input_root = benchmark_root / "dataset/task_inputs"
    task_path = Path(args.task).resolve()
    selector = args.benchmark
    if task_path.is_relative_to(input_root):
        selector = task_path.relative_to(input_root).as_posix()
    if selector:
        asyncio.run(generate_benchmark_tasks(args, benchmark_root, selector, logger))
        return
    
    # Setup task directory / task name
    # Normal模式下：--task 是任务名或任务目录；Resume模式下：--task 仅作为逻辑 task_name 使用，

    if args.resume_session:
        # Resume 时不依赖 tasks 目录，直接把传入的 task 当作 task_name，仅用于 results/{task_name}/...
        args.task_name = args.task
        args.task_dir = osp.join("results", args.task_name)
    else:
        # 如果 --task 是路径（包含 / 或 \），则直接使用；否则认为是 tasks/{task} 目录
        if '/' in args.task or '\\' in args.task or osp.isdir(args.task):
            args.task_dir = args.task
            args.task_name = osp.basename(args.task.rstrip('/\\'))
        else:
            args.task_dir = osp.join("tasks", args.task)
            args.task_name = args.task
        
        if not osp.exists(args.task_dir):
            raise FileNotFoundError(f"Task directory not found: {args.task_dir}")
    
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = osp.join("results", args.task_name)
    else:
        args.output_dir = osp.join("results", args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("DataGenHyp Geography Discovery Pipeline Started")
    logger.info(f"Task: {args.task_name}")
    logger.info(f"Task Directory: {args.task_dir}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info("=" * 80)
    
    # Step 1: Finding Generation or Session Resume
    if args.resume_session:
        # ---------------------------------------------------------------
        # Path A: resume an existing session (do NOT create a new one)
        # ---------------------------------------------------------------
        logger.info(f"Resuming existing MAS session: {args.resume_session}")

        interface = DataGenHypInterface(
            config_path=args.config,
            config=None,
            work_dir=args.output_dir,
            task_name=args.task_name,
        )

        async def _resume():
            await interface.startup()
            status_before = await interface.get_session_status(args.resume_session)
            logger.info(
                f"Session {args.resume_session} BEFORE resume: "
                f"state={status_before.get('state')}, "
                f"iterations_completed={status_before.get('iterations_completed')}"
            )
            status_after = await interface.resume_session(args.resume_session)
            logger.info(
                f"Session {args.resume_session} AFTER resume: "
                f"state={status_after.get('state')}, "
                f"iterations_completed={status_after.get('iterations_completed')}"
            )
            await interface.shutdown()
            return status_after

        try:
            status_after = asyncio.run(_resume())
        except Exception as e:
            logger.error(f"Resuming session {args.resume_session} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # When resuming we don't re-export findings.json here; user can inspect
        # results/{task_name}/traj_{session_id}/ for updated artifacts.
        top_findings = []
        session_json = None
        logger.info(
            f"Session {args.resume_session} resume complete with state={status_after.get('state')}"
        )
    else:
        # ---------------------------------------------------------------
        # Path B: standard finding generation (original behavior)
        # ---------------------------------------------------------------
        logger.info("Starting geography finding generation with MAS...")
        finding_generator = FindingGenerator(args, logger)
        
        try:
            top_findings, session_json = asyncio.run(finding_generator.generate_findings())
        except Exception as e:
            logger.error(f"Finding generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Save findings in standard format
        findings_output = osp.join(args.output_dir, "findings.json")
        os.makedirs(osp.dirname(findings_output), exist_ok=True)
        
        with open(findings_output, 'w') as f:
            json.dump(top_findings, f, indent=4)
        
        logger.info(f"Findings saved to {findings_output}")
    
    # Step 2: Summary
    logger.info("=" * 80)
    logger.info("DataGenHyp Geography Discovery Run Completed")
    logger.info("=" * 80)
    
    logger.info(f"Total findings generated: {len(top_findings)}")
    
    # Print detailed results
    logger.info("\nTop Findings:")
    for i, finding in enumerate(top_findings, 1):
        finding_id = finding.get('id', f'finding_{i}')
        finding_text = finding.get('text', finding.get('description', 'N/A'))
        finding_score = finding.get('score', 0.0)
        logger.info(f"  {i}. [{finding_id}] Score: {finding_score:.2f}")
        logger.info(f"     {finding_text[:100]}...")
    
    # Load config for summary
    config = {}
    if args.config and osp.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                if args.config.endswith(('.yaml', '.yml')):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    
    # Determine findings source path:
    # - Path B (new run): use session_json returned by FindingGenerator
    # - Path A (resume): build path to existing traj_{session_id}.json
    if args.resume_session:
        findings_source = osp.join(
            "results",
            args.task_name,
            f"traj_{args.resume_session}",
            f"traj_{args.resume_session}.json",
        )
    else:
        findings_source = session_json
    session_id=args.resume_session if args.resume_session else finding_generator.session_id
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'task': args.task_name,
        'task_dir': args.task_dir,
        'output_dir': os.path.join(args.output_dir, f"traj_{session_id}"),
        'findings_source': findings_source,
        'model': (
            config.get("agents", {}).get("proposal", {}).get("model_provider") or
            config.get("models", {}).get("default_provider", "openai/claude-sonnet-4-5-20250929")
        ),
        'total_findings': len(top_findings),
        'top_findings': top_findings
    }
    
    summary_path = osp.join(args.output_dir, "discovery_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"\nSummary saved to {summary_path}")
    logger.info("=" * 80)
    logger.info("All done!")


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiscovery pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
