# 行为生成与 LLM 分工审计

本文专门澄清 LabWars 当前的行为生成机制，避免把系统误读成“LLM 自由写剧情”或“纯手写规则触发”。

结论先说清楚：

> LabWars 当前是 **continuous latent action field proposes candidates -> LLM scores candidate plausibility -> fused sampler selects primary action -> LLM renders public/private stance and memory-compatible narration**。

这不是 LLM 自由行动系统，也不是旧式阈值规则系统，而是一种可复现、可消融、可审计的 hybrid social simulation。

---

## 1. 当前真实行为链路

实现位置：

| 环节 | 文件 |
|---|---|
| 候选动作生成 | `src/engine/action_selection.py` |
| LLM 候选评分与融合采样 | `src/engine/role_policy.py` |
| scoring/rendering prompt | `src/engine/prompts.py` |
| LLM drift 审计 | `src/engine/critic.py` |
| action-field 参数 | `config/action_field.yaml` |
| λ lesion | `python -m src.experiments paper --include-lambda` |

真实流程是：

```text
agent latent state + event + recall + relationship + project state
  -> generate_action_candidates()                    # structural field prior
  -> LLM scores each candidate plausibility           # subjective model layer
  -> fused_tendency = field_tendency + mix*(llm_score - 0.5)
  -> softmax(fused_tendency) + seeded sampling
  -> selected primary_action
  -> LLM renders public_position / private_intent / wording
  -> RolePolicyAgent overwrites raw.primary_action with selected action
  -> Critic checks wording/action drift
```

代码上最关键的是：

```python
raw["primary_action"] = {
    "type": selected_payload["type"],
    "target": selected_payload["target"],
    "intensity": selected_payload["intensity"],
}
```

也就是说，LLM 参与评分，但不能在 rendering 阶段把 `confront` 偷换成 `comply`，也不能把 `document_contribution` 偷换成 `ask_for_authorship`。

---

## 2. LLM 做什么，不做什么

| 层 | 是否由 LLM 决定 | 说明 |
|---|---:|---|
| action candidate set | 否 | 由 continuous action field 从 allowed action space 生成 |
| candidate plausibility score | 是 | LLM 根据状态、记忆、关系、事件给每个候选动作打分 |
| primary action type | 共同决定 | field prior + LLM score 融合后采样 |
| action target | 主要由 field 决定 | selected candidate 固定，normalizer 只做合法性修正 |
| action intensity | 主要由 field 决定 | tendency -> intensity curve 生成 |
| public_position | 是 | LLM 生成公开立场 |
| private_intent | 是 | LLM 生成私下目标/策略描述 |
| memory interpretation | 是 | LLM 生成第一人称主观记忆文本 |
| 数值状态更新 | 否 | 情绪、信念、关系、ledger 由动力学函数更新 |
| 事件概率 | 否 | 由 state-driven event field 计算 |

准确表述是：

```text
LLM is a candidate plausibility scorer plus interpretation/stance renderer, not an unconstrained behavioral policy.
```

---

## 3. 为什么不让 LLM 完全自由选 action？

LLM 直接选 action 的优点是 open-ended，但风险很大：

| 风险 | 影响 |
|---|---|
| 不可复现 | 同一状态不同调用可能给出完全不同动作 |
| 容易剧情化 | LLM 倾向写戏剧性冲突，而不是连续社会动力学 |
| 难做反事实 | 无法判断结果来自状态变量还是 prompt 风格 |
| 难做消融 | 无法单独移除某个 motive weight 或 field prior |
| 难审计 | 行为来源不容易反编译 |

所以当前方案选择中间路线：

```text
可复现 continuous policy prior + LLM subjective plausibility scoring + LLM narration
```

这让系统可以回答：

```text
如果删除 PI 承诺记忆，LLM score 和 fused probability 怎么变？
如果把 confront 对 resentment_drive 的权重置零，署名抗议还会出现吗？
如果关闭 LLM scoring，只用 field prior，行为轨迹差多少？
```

---

## 4. Action Field 不是阈值法，但仍是参数化 baseline

`action_selection.py` 已经不是：

```python
if resentment > 0.7:
    action = "confront"
```

而是：

```text
field_tendency = Σ motive_i * weight_i + event_affinity + noise + repetition_effect
llm_score = LLM(candidate, state, memory, relationship, event)
fused_tendency = field_tendency + mix*(llm_score - 0.5)
probability = softmax(fused_tendency / temperature)
selected_action = sample(probability)
```

但 motive weights 仍然是结构先验，不应该被宣传成“完全无启发式”。更准确地说：

```text
calibratable social-psychological field prior
+ LLM candidate plausibility scoring
+ continuous sampling
+ LLM interpretation layer
```

---

## 5. Motive weights 是 baseline/prior，不是最终真理

当前 motive weights 外置到：

```text
config/action_field.yaml
```

其中包括：

```yaml
action_field:
  baseline_tendency: 0.10
  repetition_penalty: 0.045
  temperature_base: 0.20
  temperature_arousal_scale: 0.22

motive_weights:
  confront:
    resentment_drive: 0.35
    authorship_anxiety: 0.25
    authority_pressure: 0.10
    caution: -0.35
```

每个 action log 会保留：

```json
{
  "action_candidates": [...],
  "selected_action": {
    "type": "confront",
    "field_probability": 0.184,
    "llm_score": 0.720,
    "fused_tendency": 0.558,
    "scoring_source": "field_llm_fused"
  },
  "llm_action_scoring": {
    "enabled": true,
    "source": "field_llm_fused",
    "mix": 0.35
  }
}
```

这样可以区分：

| 来源 | 含义 |
|---|---|
| `field_probability` | 手写结构场给出的 baseline 概率 |
| `llm_score` | LLM 认为该候选动作对当前 agent 是否心理合理 |
| `fused_tendency` | 融合后的采样倾向 |
| `scoring_source` | 当前是 field-only 还是 field+LLM |

---

## 6. 如何做消融？

可以分别消融两层：

| 消融 | 做法 | 看什么 |
|---|---|---|
| field weight ablation | override `config/action_field.yaml` 中某个 motive weight | 冲突是否由手写 prior 主导 |
| LLM scoring ablation | `SimConfig(enable_llm_action_scoring=False)` | LLM 主观评分是否改变轨迹 |
| mix sensitivity | 调 `llm_action_score_mix` | 系统对 LLM 评分权重是否敏感 |
| memory delete | 删除 promise/authorship memory | 长程记忆是否中介 R52 escalation |

---

## 7. 如何描述当前系统才准确？

不准确：

```text
LLM agents freely decide actions based on long-term memory.
```

更准确：

```text
LabWars uses continuous latent action fields as a structural behavioral prior, asks LLMs to score candidate action plausibility under subjective memory and relationship context, then samples actions from the fused field. LLMs also generate memory interpretations and public/private stance under the selected-action constraint.
```

中文：

```text
LabWars 使用连续 latent action field 作为结构性行为先验，让 LLM 在主观记忆和关系语境下对候选动作评分，再从融合后的行为场中采样真实行动。LLM 同时在 selected action 约束下生成记忆解释、公开立场和私下意图。
```

---

## 8. 现在比旧版强在哪里？

旧版：

```text
field samples action -> LLM explains
```

新版：

```text
field proposes candidates -> LLM scores -> fused sampler selects -> LLM explains
```

关键提升是：LLM 不再只是“包装动作”，它会影响候选动作概率；但它仍不能绕过 action space、状态审计和因果可复现性。

---

## 9. 最终边界

LabWars 当前最强的地方不是“LLM 自己演出了内斗”，而是：

```text
它把科研内斗拆成可观测、可评分、可干预、可消融、可反事实比较的社会动力学变量。
```

这比单纯让 LLM 写剧情更适合做研究平台，也比纯手写权重更接近主观社会行为仿真。
## 10. 防误读补强：legacy sampler、rank shift、mix ablation

为避免评估者误以为主流程仍是 field-only sampling，`sample_action_candidate` 已从当前 API 中移除，历史函数改名为：

```text
sample_action_candidate_legacy
```

并新增测试保证 `RolePolicyAgent` 不 import、不调用 legacy sampler。当前主流程只使用：

```text
generate_action_candidates -> LLM candidate scoring -> fused sampler
```

报告现在会输出 LLM scoring 的可视化审计：

| 字段 | 含义 |
|---|---|
| `field_top3` | structural field prior 排名前三 |
| `llm_top3` | LLM plausibility score 排名前三 |
| `fused_top3` | 融合后 tendency 排名前三 |
| `selected_action` | 最终采样动作 |
| `LLM Override Pressure` | field ranking 到 fused ranking 的 rank-shift 压力 |
| `selected_rank_lift` | selected action 在 fused ranking 中相对 field ranking 提升了多少位 |

λ 由 CausalOp `set_policy_lambda` 切换，不是 mix 扫描矩阵：

```text
λ = 0.0   # field only
λ = 0.35  # default dual_engine
λ = 1.0   # LLM-heavy (cache miss)
```

默认比较：

```text
authorship_dispute_index
trust_fragmentation
public_private_divergence_mean
memory_authorship_cluster_strength
protest_authorship
integrity_risk
llm_override_pressure
```

这个实验回答的问题是：LLM candidate scoring 是否真的改变长程社会轨迹，而不只是给 field prior 做解释包装。
## 11. 双引擎 LabWars：结构驱动 vs 语言驱动

LabWars 现在可以明确表述为双引擎架构：

| 引擎 | 名称 | 作用 |
|---|---|---|
| Engine A | Social Physics Engine | 负责记忆共振、关系图、资源压力、署名账本、event field、action candidates |
| Engine B | LLM Cognitive Policy Layer | 负责候选动作主观合理性评分、公开/私下立场、主观记忆解释 |

核心实验参数是：

```text
λ = cognitive_policy_lambda ∈ [0, 1]
```

语义：

| λ | 模式 | 含义 |
|---:|---|---|
| 0.0 | physics only | 只使用 Social Physics Engine 的候选动作排序 |
| 0.2 | weak LLM | LLM 轻微影响候选排序 |
| 0.35 | hybrid default | 当前默认双引擎平衡点 |
| 0.6 | LLM-heavy | LLM 主观评分明显改变行为分布 |
| 1.0 | LLM dominated | 候选空间仍由 physics 给出，但排序由 LLM Cognitive Policy Layer 主导 |

当前融合公式：

```text
social_physics_tendency = action field tendency
llm_cognitive_tendency = 2 * (llm_score - 0.5)
fused_tendency = (1 - λ) * social_physics_tendency + λ * llm_cognitive_tendency
probability = softmax(fused_tendency)
```

这样 LabWars 可以回答一个更硬的问题：

```text
同一批 agent、记忆和事件压力下，学术内斗轨迹主要来自结构性社会压力，还是来自 LLM 对情境的语言认知解释？
```

对应实验接口是 Causal MRI 的 λ 补丁，不是独立扫描矩阵：

```python
from src.engine.causal.algebra import set_policy_lambda
from src.engine.causal.twin import run_twin

twin = run_twin(cfg, [set_policy_lambda(1.0)], llm_trace=factual.llm_cache)
```

它会比较：

```text
authorship_dispute_index
trust_fragmentation
public_private_divergence_mean
memory_authorship_cluster_strength
protest_authorship
integrity_risk
llm_override_pressure
```

这使 LabWars 的研究问题从“能不能模拟科研内斗”升级为：

```text
社会冲突中，结构压力和语言认知分别贡献了多少？
```
## 12. 9.5 分补强：解释力、语言足迹、相变链

当前新增三类审计对象，让行为不只是“被采样出来”，而是可以反编译来源。

### 12.1 Action Field 可解释性

每个 `ActionCandidate` 现在记录 `field_decomposition`：

```json
{
  "baseline": 0.10,
  "motive_contributions": {
    "resentment_drive": 0.21,
    "credit_capture": 0.18,
    "caution": -0.09
  },
  "event_affinity": 0.08,
  "repetition_penalty": -0.045,
  "memory_trigger": {
    "content_type": "promise_broken",
    "attention": 0.42
  },
  "raw_tendency": 0.48,
  "final_tendency": 0.48
}
```

报告第 12 节会输出 selected action 的正向力、负向力和触发记忆，用来回答：

```text
为什么这个 action 从社会物理场里冒出来？
```

### 12.2 LLM Influence Footprint

除了 `field -> fused` 的 LLM Override Pressure，现在还记录：

```text
field ranking -> LLM perceived ranking
```

也就是 LLM 在“主观意图判断”层面到底把哪些 action 提前了、压低了。报告第 13 节输出：

```text
field_rank
llm_rank
fused_rank
field_to_llm_pressure
selected_llm_rank_lift
private_strategy
```

这回答：

```text
LLM 到底改写了多少行为解释和候选排序？
```

### 12.3 轨迹级因果链：从节点到相变

Path-level causal chain 现在不只是：

```text
event -> memory -> action -> outcome
```

而是显式组织成：

```text
event chain
-> memory cluster formation
-> behavioral drift
-> phase transition
```

`phase_transition_round` 由 memory increment、behavioral drift、authorship dispute、trust fragmentation 的连续压力峰值给出。它不是触发规则，只是事后反编译指标，用来解释“冲突什么时候从潜伏变成显性”。
## 13. Policy Mode 三轨对照：冲 9.5 的诚实边界

LabWars 不应对外宣称“完全无启发式”或“LLM 自由生成真实社会”。更稳的定位是：

```text
可校准的连续社会动力学先验
+ LLM 候选动作认知评分
+ 反事实消融审计
```

为了检验这个边界，当前新增三轨 `policy_mode`：

| mode | action space 来源 | action ranking 来源 | 研究用途 |
|---|---|---|---|
| `social_physics` | Action Field | Social Physics only | 结构压力 baseline |
| `dual_engine` | Action Field | Social Physics + LLM Cognitive scoring | 默认主线，可控可审计 |
| `llm_native` | LLM 直接提出候选动作，再由系统映射/校验 | LLM-native plausibility | 对照 LLM 自由生成候选是否更戏剧化、更不稳定 |

`llm_native` 不是默认主仿真，而是对照轨道。它回答：

```text
如果让 LLM 自己提出行动候选，轨迹是否更 prompt-sensitive？
是否更戏剧化？
是否比 dual_engine 更难反事实？
```

三种 `policy_mode` 由 `SimConfig` 切换。论文默认 `dual_engine`；`social_physics` 与 `llm_native` 是机制对照，不是规模扫描。

这使项目的核心问题变成：

```text
社会冲突轨迹到底来自结构性压力、语言模型认知，还是两者混合？
```
