# Tasks 分类说明

本目录包含 100 个地理空间基础任务，按地理学主题分为 7 类。每个任务绑定一个明确的底层数据资产或外源查询，不因分类或分析目标不同而为同一资产重复创建任务。

## 数据来源

任务数据可来自以下后端，`prompt.json` 中 `data_path` 的来源前缀决定实际访问方式：

| 前缀 | 访问后端 | 任务数 |
|---|---|---:|
| `GEE:` | Google Earth Engine | 90 |
| `NASA_CMR:` | NASA CMR STAC API | 3 |
| `PLANETARY_COMPUTER:` | Microsoft Planetary Computer STAC API | 2 |
| `EARTH_SEARCH:` | Element 84 Earth Search STAC API | 2 |
| `OSM_OVERPASS:` | OpenStreetMap Overpass API | 2 |
| 无在线来源前缀 | 本地文件或目录 | 1 |

外源任务在 `data_description.source_metadata` 中记录 collection URL、数据提供方或 Overpass QL。系统保留真实来源前缀，不会将外源数据改写为 GEE 资产。

## 外源与 GEE 重合约束

外源任务与现有 90 个 GEE 任务按“同一产品/同一版本/同一要素集合”判定重合，而不是按分析主题或上游卫星是否相同判定。当前复核结果为 0 个同产品冲突：Daymet、3DEP、Sentinel-2、Copernicus DEM、NASA ABoVE/ICESat-2/LVIS 均没有对应的 GEE 产品任务；OSM 查询也避开了 GEE Healthsites 医疗设施、OSM Water 图层和 TIGER 道路图层，分别使用北京公共交通站点与新加坡主干道路。

## 分类统计

| 类别 | 任务数 |
|---|---:|
| `climate_surface_interaction` | 18 |
| `environmental_monitoring` | 18 |
| `geological_hazards` | 15 |
| `geomorphology_analysis` | 14 |
| `land_use_land_cover` | 12 |
| `remote_sensing_geology` | 8 |
| `socioeconomic_geography` | 15 |
| **合计** | **100** |

## 使用说明

每个任务目录包含一个 `prompt.json`，其中描述数据集身份、时空覆盖、数据类型、分析能力以及访问来源。

```bash
python launch_discovery.py --task <类别目录>/<任务目录> --config config/debug_config.yaml
```

例如：

```bash
python launch_discovery.py --task climate_surface_interaction/Planetary_Computer_Daymet_Annual_North_America --config config/debug_config.yaml
```

分类版本：v3.2（2026-08-28，按唯一底层资产去重）。

## 外源访问检查

以下命令只查询 collection 元数据、最多一个 STAC item（无 item 的 datacube 检查集合级云资产声明）或受限的 Overpass 结果，不下载数据文件：

```bash
python scripts/check_external_task_sources.py --report artifacts/external_source_check.json
```

所有外源任务均可访问且返回数据时退出码为 `0`；声明错误时为 `2`；任一远程 collection、item 查询或 Overpass 查询失败时为 `1`。可重复使用 `--source NASA_CMR` 等参数只检查指定来源。
