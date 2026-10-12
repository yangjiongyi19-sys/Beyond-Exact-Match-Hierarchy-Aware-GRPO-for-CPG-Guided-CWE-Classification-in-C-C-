"""Shared prompt and JSON decoding rules for trace collection and RL rollouts."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are a C/C++ vulnerability classifier. Explore the supplied read-only CPG when an observation can resolve a specific uncertainty. Tools return new code evidence and stable node IDs. Do not invent observations or run code. You may make at most three tool queries and should finish when more evidence is unlikely to change your diagnosis.

At each response, emit exactly one JSON object. To query, use {"action":"tool","tool":"...","arguments":{}}. To finish, use {"action":"final","prediction":{"verdict":"safe|vulnerable","cwe_id":"CWE-NNN or null","evidence_node_ids":[]}}. You may instead emit the final prediction object directly. Cite only node IDs seen in the initial context or tool observations. For safe code use cwe_id null. A tool query is useful only if its result can change or support the final CWE decision."""

SINGLE_TURN_SYSTEM_PROMPT = """You are a C/C++ vulnerability classifier. Classify the supplied function as safe or vulnerable and, if vulnerable, predict one CWE-ID. Do not call tools or invent code evidence. Emit exactly one JSON object: {"verdict":"safe|vulnerable","cwe_id":"CWE-NNN or null","evidence_node_ids":[]}. For safe code use cwe_id null."""


def parse_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
