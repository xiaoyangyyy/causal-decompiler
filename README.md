# LabWars

> **LabWars 不模拟科研成功，而是反编译科研合作为什么变成内斗。**

一个 14 角色、60 轮的学术实验室权力博弈沙盒。论文对象是对**一条冻结事实轨迹**做 Causal MRI：冻结外生噪声和 LLM 文本，修补展开后的 SCM，读公开 vs 私下结果向量。

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

## 论文实验

不要跑独立 seed 的条件矩阵。主命令是 `paper`：identity twin、split-Y、记忆 IRF、Shapley vs skip、三世界；可选 A/B/C/D/V 的 CRN 对照。

```powershell
python -m pytest tests/ -q

# 脚本烟雾（无 API）
python -m src.experiments paper --rounds 8 --seed 11 --llm-provider scripted

# 付费 LLM：先落一条 60 轮事实轨迹，再从 jsonl 做 MRI（不重付事实世界）
$env:DEEPSEEK_API_KEY = Read-Host "Paste DEEPSEEK_API_KEY"
$env:LABWARS_LLM_CONFIG = "config/llm.deepseek.yaml"
python -m src.experiments paper --rounds 60 --full-cast --llm-provider openai --sampled-top-k 1
python -m src.experiments paper --from-jsonl output/runs/run_XXXX.jsonl --full-cast --contrasts A --contrast-seeds 0,1,2
```

`--sampled-top-k` 默认为 1（每轮只让一名 agent 走 LLM 打分）。`--from-jsonl` 回放时沿用落盘配置，不要改 top-k。`--include-lambda` 会改写 prompt，缓存会 miss。

DeepSeek 配置见 [`config/llm.deepseek.yaml`](config/llm.deepseek.yaml)（`thinking: disabled`）。密钥只放环境变量或 gitignored `.env`。

调试单条件（不是论文 MRI）：

```powershell
python -m src.experiments run -e A -c A2 --seed 42
python -m src.experiments report -e A -c A2 --seed 42
```

## LLM 分工

连续 action field 生成候选；LLM 对候选做主观 plausibility 打分；系统融合 `field_score` 与 `llm_score` 后采样 primary action。LLM 还写 memory interpretation 与公开/私下立场。LLM 不自由覆盖 primary action。

| mode | 行为生成 |
|---|---|
| `social_physics` | 只使用社会动力学先验 |
| `dual_engine` | field 候选 + LLM 打分（默认） |
| `llm_native` | LLM 直接生成候选，再映射到 action schema |

λ lesion（field vs LLM）是 MRI 可选补丁，不是单独的扫描矩阵。

## 项目结构

```
LabWars/
├── 00-总文档.md
├── README.md
├── docs/
├── schemas/
├── config/
├── src/
│   ├── world/
│   ├── cognition/
│   ├── engine/          # 含 src/engine/causal（Decompiler）
│   └── experiments/     # paper MRI, CRN contrasts, A–D/V catalog
└── output/
```
