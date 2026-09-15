# Causal Decompiler

> 把一条 agent 轨迹当成可执行的因果程序：先原样重放，再对内部构件做 `do()`，反编译成最小事件、交互与中介路径。
>
> From “Why did the agent say this?” to “What had to happen for this outcome to occur?”

LabWars 只是其中一个场景（14 角色、60 轮学术实验室权力博弈）。另外两个独立包是 CrisisGrid（空间通讯）和 ReleaseOps（工作流 DAG）。论文对象是对**一条冻结事实轨迹**做 Causal MRI：冻结外生噪声和 LLM 文本，修补展开后的 SCM，读公开 vs 私下结果向量。

主入口是 [`docs/18-Causal-Decompiler-Paper-Protocol.md`](docs/18-Causal-Decompiler-Paper-Protocol.md)。不要跑独立 seed 的条件矩阵，也不要重建已删除的 scale / 630-cell ATE 扫描。

## 快速验证

```powershell
pip install -r requirements.txt
python -m pytest tests/ -q

# 脚本烟雾（无 API）
python -m src.experiments paper --rounds 8 --seed 11 --llm-provider scripted
```

## 论文 MRI

`paper` 写出 identity twin、split-Y、记忆 IRF、Shapley vs skip、三世界。Split-Y 同时读公开抗议、潜在势、PPD，以及 idea→PI 的 `trust_pi_logged`（草案快照）和 `trust_pi_path_mean`（路径均值）。

密钥放 gitignored `.env` 的 `DEEPSEEK_API_KEY=`，不要写进 yaml。配置见 [`config/llm.deepseek.yaml`](config/llm.deepseek.yaml)（`thinking: disabled`）。

```powershell
# 连通性
python scripts/ping_deepseek.py

# 付费：一条 14 人 / 60 轮事实轨迹 + MRI
$env:LABWARS_PROGRESS = "1"
python -u -m src.experiments paper --rounds 60 --full-cast --llm-provider deepseek --sampled-top-k 1 --seed 11 --output output/reports

# 已有 jsonl 时只做 MRI，不重付事实世界
python -m src.experiments paper --from-jsonl output/runs/run_XXXX.jsonl --full-cast --output output/reports
```

`--sampled-top-k` 默认为 1。`--from-jsonl` 回放沿用落盘配置，不要改 top-k。`--include-lambda` 会改写 prompt，LLM 缓存会 miss。

产物：`output/reports/paper_protocol_{run_id}.md` 与 `.json`（目录 gitignored）。

调试单条件（不是论文 MRI）：

```powershell
python -m src.experiments run -e A -c A2 --seed 42
python -m src.experiments report -e A -c A2 --seed 42
```

`decompile` 是 `paper` 的别名。可选 `--contrasts A --contrast-seeds 0,1,2` 做 A/B/C/D/V 的 CRN 对照，不是独立 seed 网格。

## LLM 分工

连续 action field 生成候选；LLM 对候选做主观 plausibility 打分；系统融合 `field_score` 与 `llm_score` 后采样 primary action。信用类动作（`ask_for_authorship` 等）的公开立场约束为 `self_advocacy`，不覆盖为 `team_support`。LLM 还写 memory interpretation 与私下意图，不自由覆盖 primary action。

| mode | 行为生成 |
|---|---|
| `social_physics` | 只使用社会动力学先验 |
| `dual_engine` | field 候选 + LLM 打分（默认） |
| `llm_native` | LLM 直接生成候选，再映射到 action schema |

λ lesion（field vs LLM）是 MRI 可选补丁，不是单独的扫描矩阵。

## 文档索引

| 文档 | 内容 |
|------|------|
| [00-总文档.md](00-总文档.md) | 项目总览、四部分划分 |
| [docs/01-世界模型与数据结构.md](docs/01-世界模型与数据结构.md) | 14 角色、Schema、事件、动作、60 轮骨架 |
| [docs/02-认知与社会动力学.md](docs/02-认知与社会动力学.md) | 记忆/情绪/信念/关系图/署名博弈 |
| [docs/03-仿真引擎与因果干预.md](docs/03-仿真引擎与因果干预.md) | 仿真循环、多 Agent 架构、因果引擎 |
| [docs/04-实验方案与反编译报告.md](docs/04-实验方案与反编译报告.md) | 实验 A–D/V 条件目录 |
| [docs/05-连续状态驱动改造进展.md](docs/05-连续状态驱动改造进展.md) | 历史改造记录（非论文协议） |
| [docs/06-主动传递机制详解.md](docs/06-主动传递机制详解.md) | 事件/信息如何主动传递 |
| [docs/07-记忆系统设计详解.md](docs/07-记忆系统设计详解.md) | 主观记忆系统 |
| [docs/08-行为生成与LLM分工审计.md](docs/08-行为生成与LLM分工审计.md) | action field 与 LLM 分工边界 |
| [docs/09-Agent-MRI-and-Action-Field-Theory.md](docs/09-Agent-MRI-and-Action-Field-Theory.md) | Agent MRI 与 action-field |
| [docs/10-Social-Potential-Field.md](docs/10-Social-Potential-Field.md) | Social Potential Field |
| [docs/11-Agent-Social-State-Model.md](docs/11-Agent-Social-State-Model.md) | Agent Social State |
| [docs/15-Theory-Grounded-Agent-Variables.md](docs/15-Theory-Grounded-Agent-Variables.md) | 变量的社会科学锚点 |
| [docs/18-Causal-Decompiler-Paper-Protocol.md](docs/18-Causal-Decompiler-Paper-Protocol.md) | **论文实验协议（主入口）** |
| [docs/19-Causal-Decompiler-Audit.md](docs/19-Causal-Decompiler-Audit.md) | **实验 1–5 与机制审计归档** |

## 项目结构

```
causal-decompiler/
├── 00-总文档.md            # LabWars 场景的历史总览
├── README.md
├── docs/                  # 18 为论文协议；19 为审计归档
├── schemas/
├── config/                # 含 scenarios/{labwars,crisisgrid,releaseops}
├── src/
│   ├── world/
│   ├── cognition/
│   ├── engine/            # Causal Decompiler
│   └── experiments/       # paper MRI, 200 worlds, matrix
└── output/                # gitignored：runs + reports
```
