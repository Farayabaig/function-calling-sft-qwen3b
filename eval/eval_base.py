"""Score the base Qwen2.5 3B Instruct on the same 200 held out xLAM examples as the fine tuned model.

Reproduces the notebook's split (length filter with the Qwen tokenizer, examples 10,000 to 10,199),
prompt (build_chatml_prompt_only) and scoring (scoring.py). Runs on CPU with a Q8 GGUF via llama.cpp,
greedy decoding.
"""
import json, sys, time, re, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scoring import format_reward_func, tool_name_reward_func, param_key_reward_func, param_value_reward_func
from transformers import AutoTokenizer
from llama_cpp import Llama

DATA, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 200
SYSTEM_PROMPT = (
    "You are an assistant skilled at generating accurate API calls based on natural "
    "language queries. Given a user query and a list of available tools, output a structured "
    "API call with the correct endpoint and parameters in JSON format."
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

def tools_str(ex):
    tools = json.loads(ex["tools"]) if isinstance(ex["tools"], str) else ex["tools"]
    return "\n".join(f"- {t['name']}: {t['description']} | Params: {', '.join(t.get('parameters', {}).keys())}" for t in tools)

def too_long(ex):
    chat = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {ex['query']}\n\nAvailable Tools:\n{tools_str(ex)}"},
            {"role": "assistant", "content": ex["answers"]}]
    return len(tok(tok.apply_chat_template(chat, tokenize=False)).input_ids) > 1024

def prompt(ex):
    convo = [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": f"Query: {ex['query']}\n\nAvailable Tools:\n{tools_str(ex)}"},
             {"role": "assistant", "content": ""}]
    return tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

raw = json.load(open(DATA))
# The notebook filters all 60k rows, then takes rows 10,000 to 10,499 as the test split.
# Only the first ~10,250 surviving rows matter, so stop filtering once we have enough.
kept = []
for ex in raw:
    if not too_long(ex):
        kept.append(ex)
    if len(kept) >= 10000 + N:
        break
test = kept[10000:10000 + N]
print(f"test examples: {len(test)}", flush=True)

llm = Llama(model_path=MODEL, n_ctx=2048, n_threads=os.cpu_count(), verbose=False)
done = {}
if os.path.exists(OUT):
    for line in open(OUT):
        r = json.loads(line); done[r["i"]] = r
with open(OUT, "a") as f:
    for i, ex in enumerate(test):
        if i in done: continue
        t0 = time.time()
        out = llm.create_completion(prompt(ex), max_tokens=512, temperature=0.0, stop=["<|im_end|>"])
        gen = out["choices"][0]["text"]
        rec = {"i": i, "query": ex["query"], "gold": ex["answers"], "pred": gen, "secs": round(time.time() - t0, 1)}
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(f"{i+1}/{len(test)} {rec['secs']}s", flush=True)

recs = [json.loads(l) for l in open(OUT)]
recs.sort(key=lambda r: r["i"])
def score(preds, golds):
    comps = [[{"content": p}] for p in preds]
    n = len(comps)
    fm = format_reward_func(comps)
    return {"valid_json": sum(1 for x in fm if x == 0.5) / n,
            "format": sum(fm) / n,
            "tool": sum(tool_name_reward_func(comps, golds)) / n,
            "key": sum(param_key_reward_func(comps, golds)) / n,
            "value": sum(param_value_reward_func(comps, golds)) / n}
golds = [r["gold"] for r in recs]
strict = score([r["pred"] for r in recs], golds)
def unfence(s):
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    return (m.group(1) if m else s).strip()
lenient = score([unfence(r["pred"]) for r in recs], golds)
print(json.dumps({"n": len(recs), "strict": strict, "lenient_code_fences_removed": lenient}, indent=1))
