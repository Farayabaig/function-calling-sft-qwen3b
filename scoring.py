"""Scoring functions used to evaluate tool calls (copied from the notebook).

Each function takes model completions and gold answers from the xLAM dataset and returns one score per example.
Tool and parameter scores subtract a penalty for every hallucinated tool or parameter.
"""
#UPDATED 18/4/25
# Cell 1: Reward functions for Salesforce function calling dataset

from collections import Counter
import json, re

# ---------- 1. Format Reward ----------
def format_reward_func(completions, **kwargs):
    """
    ✅ Reward for producing syntactically correct structured output.

    Checks whether the model's output is:
    - Valid JSON
    - A list of objects
    - Each object has a 'name' and 'arguments' key

    🎯 Rewards:
    - +0.5 for structurally valid outputs
    - -0.2 for outputs that are not parseable JSON (strong penalty)

    🧾 Output Range: [-0.2, 0.5]
    """
    rewards = []
    for completion in completions:
        try:
            parsed = json.loads(completion[0]["content"])
            if isinstance(parsed, list) and all(
                isinstance(t, dict) and "name" in t and "arguments" in t
                for t in parsed
            ):
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        except json.JSONDecodeError:
            rewards.append(-0.2)
    return rewards

# ---------- 2. Tool Name Reward ----------
def tool_name_reward_func(completions, answer, **kwargs):
    """
    ✅ Reward for selecting correct tool/function names in model completions.

    🎯 Output Range: [0.0, 1.0]
    """
    rewards = []
    for completion, gold_str in zip(completions, answer):
        content = completion[0]["content"]
        try:
            pred = json.loads(content)
            gold = json.loads(gold_str)
            pred_tools = [t.get("name") for t in pred if isinstance(t, dict)]
            gold_tools = [t["name"] for t in gold]

            pred_counts = Counter(pred_tools)
            gold_counts = Counter(gold_tools)
            matched = sum(min(pred_counts[t], gold_counts[t]) for t in gold_counts)
            base_score = matched / len(gold_tools) if gold_tools else 0.0

            hallucinated = [t for t in pred_tools if t not in gold_tools]
            penalty = 0.2 * len(hallucinated)
            final = max(0.0, base_score - penalty)
            rewards.append(final)
        except Exception:
            gold = json.loads(gold_str)
            gold_tools = list({t["name"] for t in gold})
            found = [
                tool for tool in gold_tools
                if re.search(rf"\b{re.escape(tool.lower())}\b", content.lower())
            ]
            rewards.append(len(found) / len(gold_tools) if gold_tools else 0.0)
    return rewards

# ---------- 3. Parameter Key Reward ----------
def param_key_reward_func(completions, answer, **kwargs):
    """
    ✅ Reward for selecting correct parameter keys in tool arguments.

    🎯 Output Range: [0.0, 1.0]
    """
    rewards = []
    for completion, gold_str in zip(completions, answer):
        content = completion[0]["content"]
        try:
            pred = json.loads(content)
            gold = json.loads(gold_str)
            pred_keys = [
                key for t in pred if isinstance(t, dict)
                for key in t.get("arguments", {}).keys()
            ]
            gold_keys = [
                key for g in gold
                for key in g["arguments"].keys()
            ]

            pred_counts = Counter(pred_keys)
            gold_counts = Counter(gold_keys)
            matched = sum(min(pred_counts[k], gold_counts[k]) for k in gold_counts)
            base_score = matched / len(gold_keys) if gold_keys else 0.0

            hallucinated = [k for k in pred_keys if k not in gold_keys]
            penalty = 0.1 * len(hallucinated)
            final = max(0.0, base_score - penalty)
            rewards.append(final)
        except Exception:
            gold = json.loads(gold_str)
            gold_keys = list({k for g in gold for k in g["arguments"].keys()})
            found = [
                k for k in gold_keys
                if re.search(rf"\b{re.escape(k.lower())}\b", content.lower())
            ]
            rewards.append(len(found) / len(gold_keys) if gold_keys else 0.0)
    return rewards

# ---------- 4. Parameter Value Reward ----------
def param_value_reward_func(completions, answer, **kwargs):
    """
    ✅ Reward for predicting correct parameter values.

    🎯 Output Range: [0.0, 1.0]
    """
    rewards = []
    for completion, gold_str in zip(completions, answer):
        try:
            pred = json.loads(completion[0]["content"])
            gold = json.loads(gold_str)
            pred_vals = [
                str(v).strip().lower()
                for t in pred if isinstance(t, dict)
                for v in t.get("arguments", {}).values()
            ]
            gold_vals = [
                str(v).strip().lower()
                for g in gold
                for v in g["arguments"].values()
            ]
            correct = len(set(pred_vals) & set(gold_vals))
            total = len(set(gold_vals))
            rewards.append(correct / total if total > 0 else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

# ---------- 5. Aggregated Reward ----------
def aggregated_reward_func(completions, answer, **kwargs):
    """
    ✅ Combines all the other reward functions into a single scalar score.
    🧾 Output Range: [-0.2, ~3.0]
    """
    f_scores = format_reward_func(completions)
    t_scores = tool_name_reward_func(completions, answer)
    k_scores = param_key_reward_func(completions, answer)
    v_scores = param_value_reward_func(completions, answer)
    return [f + t + 0.5*k + v for f, t, k, v in zip(f_scores, t_scores, k_scores, v_scores)]
