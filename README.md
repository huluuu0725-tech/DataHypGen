# DataHypGen

DataHypGen is a multi-agent system for generating research hypotheses from geospatial data. Starting with one or more datasets, it uses dataset profiles, published evidence, and several lines of questioning to refine hypotheses, check the resources they require, and produce structured research plans.

## Method overview

![Overview of the DataHypGen method](docs/assets/method_overview.png)

[View the vector diagram (PDF)](docs/assets/method_overview.pdf)

1. **Generate ideas from the data.** The system profiles variable definitions, spatial and temporal coverage, resolution, and data quality limitations. The profile guides the literature search and close reading of relevant papers so that candidate ideas draw on both the data and published evidence.
2. **Refine ideas through questioning.** Across several lines of inquiry, agents question, defend, and judge each idea’s scientific basis and research value. They examine the proposed mechanism, whether the analytical steps are consistent with it, the evidence for each claim, novelty, and the potential for discovery. The system retains the evidence and judgments from each round.
3. **Check resources and design the study.** The system matches research requirements to specific data products, verifies access and availability, and refines the data configuration, analysis steps, validation methods, and conditions that would falsify the hypothesis. Users carry out the experiments described in the resulting plan.
4. **Evaluate proposals independently.** The benchmark uses reference sets and a knowledge graph to report separate scores for Alignment, Depth, and resource feasibility (ResourceCheck).

## Project structure

| Path | Contents |
| --- | --- |
| [launch_discovery.py](launch_discovery.py) | Entry point for generating research plans and resuming existing sessions |
| [datahypgen/](datahypgen/) | Multi-agent workflow, research memory, context management, and tools for working with data and literature |
| [config/](config/) | Model, agent, and tool settings. The default configuration is `debug_config.yaml`. |
| [tasks/](tasks/) | Ready-to-run geospatial task inputs |
| [benchmark/](benchmark/) | Benchmark tasks, reference sets, evaluation rules, and commands for individual evaluation stages |
| [requirements.txt](requirements.txt) | Python dependencies |

## Installation and configuration

From the project root, create and activate the `datahypgen` Conda environment, then install the dependencies.

```bash
conda create -n datahypgen python=3.11 -y
conda activate datahypgen
python -m pip install -r requirements.txt
```

Configure each agent’s model and parameters in [config/debug_config.yaml](config/debug_config.yaml). This file is the default for both standard runs and benchmark generation.

The launcher reads `.env` from the project root. Set the service URLs and credentials for the models you configured.

```dotenv
OPENAI_API_KEY=<your OpenAI-compatible service API key>
OPENAI_API_BASE_URL=<your OpenAI-compatible service URL>
QWEN_API_KEY=<your Qwen service API key>
QWEN_API_BASE_URL=<your Qwen service URL>
```

Configure access for each task’s data sources. Google Earth Engine (GEE) data requires valid Earth Engine authorization. Choose model names supported by your provider. Keep API keys and service account files in your local environment or outside the repository.

## Generate research plans

DataHypGen includes 100 geospatial tasks in `tasks/` and 101 benchmark tasks in `benchmark/dataset/task_inputs/`. The benchmark has 49 single-dataset tasks, 32 two-dataset tasks, and 20 tasks that use three or more datasets. You can also provide a custom task directory.

### Built-in and custom tasks

Run all commands from the project root. To generate a plan for the North American Daymet task, run

```bash
python launch_discovery.py \
  --task tasks/climate_surface_interaction/Planetary_Computer_Daymet_Annual_North_America \
  --config config/debug_config.yaml
```

Pass the task directory to `--task`. If you supply only a directory name, the launcher looks for `tasks/<directory_name>`. For tasks nested under a category, pass the full relative path as shown above.

For a custom task, create a directory containing a `prompt.json` file that describes the research objective, input data, and where to access the data. See the [task guide](tasks/README.md) for the format and the [benchmark task input contract](benchmark/protocol/task_input_contract.md) for benchmark-specific requirements.

### Benchmark tasks

Generate plans for all 101 benchmark tasks with

```bash
python launch_discovery.py --benchmark all --config config/debug_config.yaml
```

To generate plans for a single benchmark task, use its canonical ID.

```bash
python launch_discovery.py \
  --benchmark single_dataset/task_001 \
  --config config/debug_config.yaml
```

The `--benchmark` option accepts a canonical ID from the [task manifest](benchmark/dataset/task_groups.json), a group name (`single_dataset`, `dual_dataset`, or `three_or_more_datasets`), or `all`. You can also specify the task directory directly with `--task benchmark/dataset/task_inputs/single_dataset/task_001`.

Standard tasks write results to `results/<task_name>/`. Benchmark tasks write to `benchmark/system_outputs/ours/<group>/<task_id>/`. Benchmark generation stops before the internal research-value assessment. Run the independent evaluation separately.

### Results and session recovery

Standard runs save `findings.json`, stage outputs, and a session trace at `traj_<session_id>/traj_<session_id>.json` under `results/<task_name>/`. The trace records the session state, research plans, and stage outputs. Logs are written to `logs/`.

To resume an existing session, supply its original task name and session ID.

```bash
python launch_discovery.py \
  --task <original_task_name> \
  --resume_session <session_id> \
  --config config/debug_config.yaml
```

## Benchmark evaluation and reproducibility

The 101 benchmark tasks cover studies using one, two, or three or more datasets. Generation reads only the task inputs. Reference studies and source papers are used during evaluation. See the [task groups](benchmark/dataset/task_groups.json) and the [benchmark guide](benchmark/README.md) for details.

To evaluate generated benchmark results on all three dimensions and produce a summary, run

```bash
python benchmark/run.py evaluate -- --axes all --systems ours
python benchmark/run.py summary
```

To evaluate a single task, specify its ID.

```bash
python benchmark/run.py evaluate -- \
  --axes all --systems ours --task-id single_dataset/task_001
```

ResourceCheck reads model, tool, and agent settings from `config/debug_config.yaml` by default. Reference retrieval and Alignment use the `semantic_retrieval` section of the same file. To override these settings, use `--resource-check-config config/config.yaml` or `--retrieval-config <config_path>`. A YAML retrieval configuration must include a `semantic_retrieval` section. A JSON configuration must provide the full set of retrieval parameters.

| Dimension | What it measures | Rules or implementation |
| --- | --- | --- |
| Alignment | Retrieves reference studies and assesses how closely a proposal matches their research directions. | [Semantic reference retrieval](benchmark/core/semantic_reference_retrieval.py) |
| Depth | Rates the research depth of a complete proposal on a scale from 1 to 10. | [Review prompt](benchmark/protocol/depth_review_prompt_v1.md) and [rubric](benchmark/protocol/depth_rubric_v1.json) |
| ResourceCheck | Checks the data and tools required to carry out the proposed study. | [ResourceCheck guide](benchmark/resource_check/README.md) |
# DataHypGen
