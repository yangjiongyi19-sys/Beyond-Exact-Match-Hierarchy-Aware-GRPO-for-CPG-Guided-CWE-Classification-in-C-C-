"""VERL experimental AgentLoop adapter for the read-only CWE CPG harness.

The import path is optional so data tooling remains usable without VERL installed.
This adapter targets VERL's experimental AgentLoop API (alpha); pin the exact git
revision used by each training run.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from .cpg_harness import CpgHarness
from .cwe_taxonomy import CweTaxonomy, canonical_cwe
from .prompts import SYSTEM_PROMPT, SINGLE_TURN_SYSTEM_PROMPT, parse_json_object
from .rewards import GroundTruth, Prediction, score_episode

try:
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
    from verl.utils.rollout_trace import rollout_trace_op
except ImportError as exc:  # pragma: no cover - cluster-only dependency
    AgentLoopBase = object  # type: ignore[assignment,misc]
    AgentLoopMetrics = None  # type: ignore[assignment]
    AgentLoopOutput = None  # type: ignore[assignment]
    register = None  # type: ignore[assignment]
    rollout_trace_op = lambda fn: fn  # type: ignore[assignment]
    _VERL_IMPORT_ERROR: Exception | None = exc
else:
    _VERL_IMPORT_ERROR = None

def _config_value(config: Any, key: str, default: Any = None) -> Any:
    try:
        return config.get(key, default)
    except AttributeError:
        return getattr(config, key, default)


if register is not None:

    @register("vulnrl_cwe_agent")
    class CweAgentLoop(AgentLoopBase):
        """Bounded multi-turn evidence gathering with deterministic terminal rewards."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            config = _config_value(self.config, "vulnrl", {})
            taxonomy_path = _config_value(config, "taxonomy_path")
            snapshot_dir = _config_value(config, "snapshot_dir")
            self.tools_enabled = bool(_config_value(config, "tools_enabled", True))
            if not taxonomy_path or (self.tools_enabled and not snapshot_dir):
                raise ValueError("VERL config needs vulnrl.taxonomy_path and, when using tools, vulnrl.snapshot_dir")
            self.taxonomy = CweTaxonomy.from_json(taxonomy_path)
            self.snapshot_dir = snapshot_dir
            self.max_cpg_queries = int(_config_value(config, "max_cpg_queries", 3)) if self.tools_enabled else 0
            self.max_tool_results = int(_config_value(config, "max_tool_results", 12))
            self.hierarchy_reward_mode = str(_config_value(config, "hierarchy_reward_mode", "alpha"))
            if self.hierarchy_reward_mode not in {"alpha", "exact"}:
                raise ValueError("vulnrl.hierarchy_reward_mode must be alpha or exact")

        @rollout_trace_op
        async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs):
            if _VERL_IMPORT_ERROR is not None:
                raise RuntimeError("VERL is required to run CweAgentLoop") from _VERL_IMPORT_ERROR
            sample_id = str(kwargs["sample_id"])
            if sample_id in {".", ".."} or "/" in sample_id or "\\" in sample_id:
                raise ValueError("sample_id must be a plain identifier for the CPG snapshot filename")
            code = str(kwargs["code"])
            truth = GroundTruth(
                is_vulnerable=bool(kwargs["is_vulnerable"]),
                cwe_id=canonical_cwe(kwargs.get("cwe_id")),
            )
            if truth.is_vulnerable and truth.cwe_id is None:
                raise ValueError(f"Training sample {sample_id} has no CWE label")

            if not self.tools_enabled:
                harness = None
                initial = None
            else:
                harness = CpgHarness(
                    self.snapshot_dir,
                    max_queries=self.max_cpg_queries,
                    max_results=self.max_tool_results,
                )
                initial = harness.reset(sample_id)
            system_content = SYSTEM_PROMPT
            user_content = "Function source:\n```c/c++\n" + code + "\n```"
            if self.tools_enabled:
                system_content += "\n\nTool schemas:\n" + json.dumps(harness.tool_schemas(), ensure_ascii=False)
                user_content += "\nInitial CPG observation:\n" + json.dumps(initial, ensure_ascii=False)
            messages = [
                {"role": "system", "content": system_content if self.tools_enabled else SINGLE_TURN_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            prompt_ids = await self.ct_build_initial_tokens(messages)
            runtime_tokens = list(prompt_ids)
            response_mask: list[int] = []
            response_logprobs: list[float] = []
            request_id = uuid4().hex
            final: Prediction | None = None
            assistant_turns = 0
            max_turns = self.max_cpg_queries + 1

            while assistant_turns < max_turns and final is None:
                assistant_turns += 1
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=runtime_tokens,
                    sampling_params=sampling_params,
                    priority=int(priority),
                )
                assistant_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
                merge_result, response_mask, aligned_logprobs = await self.ct_merge_assistant_token(
                    runtime_tokens,
                    output.token_ids,
                    response_mask,
                    response_logprobs if (response_logprobs or output.log_probs) else None,
                    assistant_logprobs=output.log_probs if output.log_probs else None,
                )
                runtime_tokens = merge_result.token_ids
                if aligned_logprobs is not None:
                    response_logprobs = aligned_logprobs
                messages.append({"role": "assistant", "content": assistant_text})
                parsed = parse_json_object(assistant_text)
                if parsed is None:
                    final = Prediction("safe", None, ())
                    break

                action = str(parsed.get("action", "final")).lower()
                if action == "final":
                    final_value = parsed.get("prediction", parsed)
                    try:
                        final = Prediction.from_dict(final_value)
                    except (ValueError, TypeError):
                        final = Prediction("safe", None, ())
                    break

                if action != "tool":
                    # Stop can carry a final prediction; otherwise ask no further tools.
                    if action == "stop" and isinstance(parsed.get("prediction"), dict):
                        try:
                            final = Prediction.from_dict(parsed["prediction"])
                        except ValueError:
                            final = Prediction("safe", None, ())
                    else:
                        final = Prediction("safe", None, ())
                    break

                if harness is None:
                    final = Prediction("safe", None, ())
                    break

                tool_name = str(parsed.get("tool", ""))
                arguments = parsed.get("arguments", {})
                if not isinstance(arguments, dict):
                    arguments = {}
                observation = harness.step(tool_name, arguments)
                previous_messages = list(messages)
                messages.append({
                    "role": "tool",
                    "content": json.dumps(observation, ensure_ascii=False, sort_keys=True),
                })
                context_merge, response_mask, aligned_logprobs = await self.ct_merge_context_msg(
                    previous_messages,
                    messages,
                    runtime_tokens,
                    response_mask,
                    response_logprobs if response_logprobs else None,
                )
                runtime_tokens = context_merge.token_ids
                if aligned_logprobs is not None:
                    response_logprobs = aligned_logprobs

            if final is None:
                final = Prediction("safe", None, ())
            reward = score_episode(
                final=final,
                truth=truth,
                taxonomy=self.taxonomy,
                observed_node_ids=harness.observed_node_ids if harness is not None else set(),
                query_count=harness.query_count if harness is not None else 0,
                max_queries=self.max_cpg_queries,
                hierarchy_mode=self.hierarchy_reward_mode,
            )
            response_ids = runtime_tokens[len(prompt_ids) : len(prompt_ids) + self.rollout_config.response_length]
            response_mask = response_mask[: self.rollout_config.response_length]
            return AgentLoopOutput(
                prompt_ids=runtime_tokens[: len(prompt_ids)],
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs[: self.rollout_config.response_length] if response_logprobs else None,
                reward_score=reward.total,
                num_turns=assistant_turns,
                metrics=AgentLoopMetrics(),
                extra_fields={
                    "sample_id": sample_id,
                    "terminal_reward": reward.to_dict(),
                    "observed_node_ids": sorted(harness.observed_node_ids) if harness is not None else [],
                    "predicted_verdict": final.verdict,
                    "predicted_cwe": final.cwe_id,
                    "predicted_evidence_node_ids": list(final.evidence_node_ids),
                },
            )

else:

    class CweAgentLoop:  # pragma: no cover - dependency guard
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Install VERL before instantiating CweAgentLoop") from _VERL_IMPORT_ERROR

