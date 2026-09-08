"""LabWars simulation engine 鈥?main loop."""

from __future__ import annotations

import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.cognition.memory import decay_memories
from src.cognition.pipeline import commit_cognition_phase, pre_decision_recall
from src.cognition.power import career_hostage_index, pi_control_pressure, pi_control_surface
from src.cognition.pressure_fields import summarize_pressure_fields
from src.engine.causal.llm_trace import LLMTrace, TracingAdapter, bind_llm_trace, current_llm_trace, reset_llm_trace
from src.engine.causal.noise import NoiseLog, bind_noise_log, reset_noise_log
from src.engine.critic import CriticAgent
from src.engine.event_agent import EventAgent, is_agent_active
from src.engine.intervention import (
    Intervention,
    apply_event_override,
    apply_world_intervention,
    get_active_interventions,
    load_interventions,
)
from src.engine.llm_adapter import LLMAdapter, QuotaExhaustedError, get_adapter, load_llm_config
from src.engine.probe import ProbeAgent
from src.engine.role_policy import RolePolicyAgent
from src.engine.run_log import RunLog, finalize_outcomes
from src.engine.story_cast import event_cast_asdict
from src.world.actions import ActionType, apply_project_effects
from src.world.loader import PROJECT_ROOT, load_world
from src.world.models import AgentRole, ProjectMetrics, WorldState
from src.world.organization import authority_ids, resolve_event_cast

CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass
class SimConfig:
    max_rounds: int = 60
    seed: int = 0
    active_agents: list[str] | None = None
    offstage_agents: list[str] | None = None
    interventions: list[Intervention] = field(default_factory=list)
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_temperature: float | None = None
    llm_adapter: LLMAdapter | None = None
    run_id: str | None = None
    output_dir: Path | None = None
    mvp: bool = False
    disable_memory: bool = False
    shuffle_memory: bool = False
    disable_state_events: bool = False
    experiment_id: str | None = None
    condition_id: str | None = None
    policy_mode: str = "dual_engine"
    enable_llm_action_scoring: bool = True
    cognitive_policy_lambda: float | None = 0.35
    llm_action_score_mix: float = 0.35
    hierarchy_lesion: bool = False
    status_lesion: bool = False
    trust_lesion: bool = False
    observation_lesion: bool = False
    cognitive_sampling_top_k: int | None = None
    cognitive_sampling_threshold: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        llm_cfg = load_llm_config()
        return {
            "max_rounds": self.max_rounds,
            "seed": self.seed,
            "active_agents": self.active_agents,
            "offstage_agents": self.offstage_agents,
            "interventions": [i.intervention_id for i in self.interventions],
            "mvp": self.mvp,
            "disable_memory": self.disable_memory,
            "shuffle_memory": self.shuffle_memory,
            "disable_state_events": self.disable_state_events,
            "experiment_id": self.experiment_id,
            "condition_id": self.condition_id,
            "policy_mode": self.policy_mode,
            "enable_llm_action_scoring": self.enable_llm_action_scoring,
            "cognitive_policy_lambda": self.cognitive_policy_lambda,
            "llm_action_score_mix": self.llm_action_score_mix,
            "hierarchy_lesion": self.hierarchy_lesion,
            "status_lesion": self.status_lesion,
            "trust_lesion": self.trust_lesion,
            "observation_lesion": self.observation_lesion,
            "cognitive_sampling_top_k": self.cognitive_sampling_top_k,
            "cognitive_sampling_threshold": self.cognitive_sampling_threshold,
            "llm_provider": self.llm_provider or llm_cfg.get("provider"),
            "llm_model": self.llm_model or llm_cfg.get("model"),
            "llm_temperature": self.llm_temperature,
        }


def load_mvp_config() -> SimConfig:
    path = CONFIG_DIR / "mvp.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    intervention_ids = data.get("default_interventions", [])
    all_interventions = {i.intervention_id: i for i in load_interventions()}
    return SimConfig(
        max_rounds=data.get("max_rounds", 20),
        active_agents=data.get("active_agents"),
        offstage_agents=data.get("offstage_agents", ["rival_lab_h"]),
        mvp=True,
        interventions=[all_interventions[iid] for iid in intervention_ids if iid in all_interventions],
    )


def _filter_world(world: WorldState, config: SimConfig) -> WorldState:
    if not config.active_agents:
        return world
    w = copy.deepcopy(world)
    keep = set(config.active_agents) | set(config.offstage_agents or [])
    w.agents = {k: v for k, v in w.agents.items() if k in keep}
    internal = [a for a in w.world_config.get("internal_agents", []) if a in keep]
    w.world_config["internal_agents"] = internal
    w.relationships = [e for e in w.relationships if e.source in keep and e.target in keep]
    return w



def _apply_hierarchy_lesion(world: WorldState) -> WorldState:
    """Flatten PI-centered authority while preserving the same event stream.

    This is a mechanism lesion for organization-level ablations: it lowers
    authority dependence, PI dependency, and funding pressure while increasing
    alternative access. It deliberately avoids deleting agents or events.
    """
    w = copy.deepcopy(world)
    for agent in w.agents.values():
        agent.personality.authority_dependence = min(agent.personality.authority_dependence, 0.18)
        agent.resources.pi_access = max(agent.resources.pi_access, 0.72)
        agent.resources.external_network = max(agent.resources.external_network, 0.68)
        agent.beliefs.pi_fairness = max(agent.beliefs.pi_fairness, 0.52)
    for edge in w.relationships:
        if edge.target in authority_ids(w) or edge.source in authority_ids(w):
            edge.dependency = min(edge.dependency, 0.18)
            edge.obligation = min(edge.obligation, 0.25)
            edge.information_access = max(edge.information_access, 0.58)
    w.project.project.funding_pressure = min(w.project.project.funding_pressure, 0.20)
    w.world_config["hierarchy_lesion"] = True
    return w



def _apply_status_lesion(world: WorldState) -> WorldState:
    """Remove status/credit-attribution incentives while preserving task pressure."""
    w = copy.deepcopy(world)
    internal = [aid for aid in w.world_config.get("internal_agents", []) if aid in w.agents]
    for agent in w.agents.values():
        agent.personality.credit_sensitivity = min(agent.personality.credit_sensitivity, 0.18)
        agent.personality.ambition = min(agent.personality.ambition, 0.48)
        agent.beliefs.my_first_author_probability = 0.35
        agent.beliefs.my_contribution_recognized = max(agent.beliefs.my_contribution_recognized, 0.78)
        agent.beliefs.others_are_free_riding = min(agent.beliefs.others_are_free_riding, 0.08)
    for edge in w.relationships:
        edge.perceived_credit_threat = min(edge.perceived_credit_threat, 0.06)
    for dimension, ledger in w.project.contribution_ledger.items():
        keys = [aid for aid in internal if aid in w.agents]
        if not keys:
            continue
        share = round(1.0 / len(keys), 6)
        w.project.contribution_ledger[dimension] = {aid: share for aid in keys}
    w.project.project.authorship_conflict = min(w.project.project.authorship_conflict, 0.06)
    w.world_config["status_lesion"] = True
    return w


def _apply_trust_lesion(world: WorldState) -> WorldState:
    """Cut trust/alliance as a state channel while leaving other social pressure intact."""
    w = copy.deepcopy(world)
    for agent in w.agents.values():
        agent.beliefs.team_trust = 0.50
    for edge in w.relationships:
        edge.trust = 0.50
        edge.resentment = 0.0
        edge.alliance = 0.0
        edge.last_interaction_valence = 0.0
    w.world_config["trust_lesion"] = True
    return w

def _shuffle_memory_refs(world: WorldState, seed: int) -> None:
    rng = random.Random(seed)
    for agent in world.agents.values():
        if len(agent.memory) < 2:
            continue
        refs = [m.get("event_ref", "E000") for m in agent.memory]
        shuffled = refs[:]
        rng.shuffle(shuffled)
        for mem, ref in zip(agent.memory, shuffled):
            mem["event_ref"] = ref


def _relationship_snapshot(world: WorldState) -> dict[str, float]:
    snap: dict[str, float] = {}
    for edge in world.relationships:
        snap[f"trust_{edge.source}_{edge.target}"] = edge.trust
        snap[f"resentment_{edge.source}_{edge.target}"] = edge.resentment
    return snap


def _resolve_llm(cfg: SimConfig) -> LLMAdapter:
    if cfg.llm_adapter is not None:
        return cfg.llm_adapter
    return get_adapter(
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
    )


def run_simulation(config: SimConfig | None = None) -> RunLog:
    cfg = config or SimConfig()
    noise = NoiseLog()
    noise_token = bind_noise_log(noise)
    trace = current_llm_trace()
    llm_token = None
    if trace is None:
        trace = LLMTrace()
        llm_token = bind_llm_trace(trace)
    try:
        return _run_simulation(cfg, noise, trace)
    finally:
        reset_noise_log(noise_token)
        if llm_token is not None:
            reset_llm_trace(llm_token)


def _seal_run_log(
    log: RunLog,
    world: WorldState,
    noise: NoiseLog,
    trace: LLMTrace,
    hits0: int,
    misses0: int,
    probe: ProbeAgent,
    cfg: SimConfig,
) -> None:
    finalize_outcomes(log, world.agents, world.relationships)
    log.outcomes["career_hostage_index"] = career_hostage_index(world)
    log.outcomes["pi_control_surface"] = pi_control_surface(world)
    log.outcomes.update(summarize_pressure_fields(log.actions))
    log.outcomes["probe_suggestions"] = probe.suggest(log.round_records)
    log.noise_log = noise.to_list()
    log.llm_cache = trace
    log.outcomes["noise_draw_count"] = len(log.noise_log)
    log.outcomes["llm_trace_stats"] = {
        **trace.stats(),
        "run_hits": trace.hits - hits0,
        "run_misses": trace.misses - misses0,
    }
    if cfg.output_dir:
        log.write_jsonl(Path(cfg.output_dir) / f"run_{log.run_id}.jsonl")


def _run_simulation(cfg: SimConfig, noise: NoiseLog, trace: LLMTrace) -> RunLog:
    if cfg.mvp:
        mvp = load_mvp_config()
        cfg.active_agents = cfg.active_agents or mvp.active_agents
        cfg.offstage_agents = cfg.offstage_agents or mvp.offstage_agents
        if cfg.max_rounds == 60:
            cfg.max_rounds = mvp.max_rounds

    run_id = cfg.run_id or str(uuid.uuid4())[:8]
    log = RunLog(run_id=run_id, config=cfg.to_dict())

    world = _filter_world(load_world(), cfg)
    if cfg.hierarchy_lesion:
        world = _apply_hierarchy_lesion(world)
    if cfg.status_lesion:
        world = _apply_status_lesion(world)
    if cfg.trust_lesion:
        world = _apply_trust_lesion(world)
    inner = _resolve_llm(cfg)
    if isinstance(inner, TracingAdapter):
        inner.trace = trace
        llm: LLMAdapter = inner
    else:
        llm = TracingAdapter(inner, trace)
    hits0, misses0 = trace.hits, trace.misses
    event_agent = EventAgent(seed=cfg.seed, state_events=not cfg.disable_state_events)
    policy = RolePolicyAgent(llm=llm)
    critic = CriticAgent()
    probe = ProbeAgent()
    event_cast = resolve_event_cast(world)
    log.config["event_cast"] = event_cast_asdict(event_cast)

    sim_config_dict = {
        "seed": cfg.seed,
        "policy_mode": cfg.policy_mode,
        "enable_llm_action_scoring": cfg.enable_llm_action_scoring,
        "cognitive_policy_lambda": cfg.cognitive_policy_lambda,
        "llm_action_score_mix": cfg.llm_action_score_mix,
        "active_agents": cfg.active_agents,
        "offstage_agents": cfg.offstage_agents,
        "offstage_min_round": 21,
        "hierarchy_lesion": cfg.hierarchy_lesion,
        "status_lesion": cfg.status_lesion,
        "trust_lesion": cfg.trust_lesion,
        "observation_lesion": cfg.observation_lesion,
        "cognitive_sampling_top_k": cfg.cognitive_sampling_top_k,
        "cognitive_sampling_threshold": cfg.cognitive_sampling_threshold,
    }

    round_num = 0
    try:
        for round_num in range(1, cfg.max_rounds + 1):
            if os.environ.get("LABWARS_PROGRESS"):
                print(f"    round {round_num}/{cfg.max_rounds} run={run_id}", flush=True)
            active_inters = get_active_interventions(cfg.interventions, round_num)

            event = event_agent.generate(round_num, world)
            if event is None:
                continue

            intervention_id = None
            for inter in active_inters:
                if inter.skip_event:
                    if not cfg.disable_memory:
                        for agent in world.agents.values():
                            decay_memories(agent, round_num)
                    log.interventions_applied.append({"round": round_num, **asdict(inter), "skipped": True})
                    event = None
                    break
                if inter.type == "memory_intervention":
                    removed = apply_world_intervention(world, inter, event_cast)
                    intervention_id = inter.intervention_id
                    entry = {"round": round_num, **asdict(inter)}
                    if removed:
                        entry["memories_removed"] = removed
                    log.interventions_applied.append(entry)
                elif inter.target_event is None or inter.target_event == event.event_id:
                    event = apply_event_override(event, inter, event_cast)
                    intervention_id = inter.intervention_id
                    log.interventions_applied.append({"round": round_num, **asdict(inter)})

            if event is None:
                continue

            if cfg.shuffle_memory:
                _shuffle_memory_refs(world, cfg.seed + round_num)

            log.record_event(event, intervention_id)

            recalls = pre_decision_recall(
                world,
                event,
                disable_memory=cfg.disable_memory,
                omniscient_observation=cfg.observation_lesion,
            )
            raw_actions = policy.decide_all(world, event, recalls, sim_config_dict)

            vetted_actions: list[dict[str, Any]] = []
            for act in raw_actions:
                agent = world.agents[act["agent"]]
                violations = critic.check(act, agent, world)
                if violations:
                    log.critic_violations.extend([
                        {"round": round_num, "agent": act["agent"], **v.__dict__} for v in violations
                    ])
                    act, _ = critic.fix_or_reject(act, agent, violations)
                vetted_actions.append(act)
                log.record_action(act["agent"], act, round_num)

            cog = commit_cognition_phase(
                world,
                event,
                recalls,
                actions=vetted_actions,
                disable_memory=cfg.disable_memory,
                llm_adapter=llm,
                omniscient_observation=cfg.observation_lesion,
            )
            if cfg.status_lesion:
                world = _apply_status_lesion(world)
            if cfg.trust_lesion:
                world = _apply_trust_lesion(world)

            for act in vetted_actions:
                try:
                    action_enum = ActionType(act["type"])
                    pd = world.project.project.model_dump()
                    pd = apply_project_effects(pd, action_enum)
                    world.project.project = ProjectMetrics(**pd)
                except ValueError:
                    continue

            if event.type == "submission_decision" and event.payload.get("submission_status") == "submitted":
                world.project.submission_status = "submitted"
            if event.type == "authorship_draft":
                world.project.author_order_draft = event.payload.get("author_order", [])

            focal = world.agents.get(event_cast.idea) or next(
                (agent for agent in world.agents.values() if agent.role == AgentRole.IDEA_ORIGINATOR),
                None,
            )
            pressure = pi_control_pressure(world, focal)
            metrics = {
                **cog.metrics,
                "career_hostage_index": career_hostage_index(world),
                "pi_control_pressure_phd_a": pressure,
                "pi_control_pressure_focal": pressure,
                "integrity_risk": world.project.project.integrity_risk,
                **_relationship_snapshot(world),
            }
            log.record_round(round_num, event.event_id, metrics, cog.agent_deltas, intervention_id)

            if world.project.submission_status == "accepted":
                break
    except QuotaExhaustedError as exc:
        log.outcomes["stopped_reason"] = "quota_exhausted"
        log.outcomes["stopped_round"] = round_num
        try:
            _seal_run_log(log, world, noise, trace, hits0, misses0, probe, cfg)
        except Exception:
            log.llm_cache = trace
        exc.partial_log = log
        raise

    _seal_run_log(log, world, noise, trace, hits0, misses0, probe, cfg)
    return log
