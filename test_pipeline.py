"""
Pipeline simulator — runs on the Pi without the GUI, microphone or speaker.
Type queries and see the full pipeline response (with per-stage timings) in the terminal.
Useful for checking Ollama and response speed independently of audio.
Press Ctrl+C to quit.
"""
import sys
import time

from database import JarvisDB
from agents import AgentPipeline
import config

print("\n" + "="*55)
print("  J.A.R.V.I.S — Pipeline Simulator")
print("="*55)
print(f"  Model  : {config.MODEL}")
print(f"  DB     : {config.DB_PATH}")
print("  Type a query and press Enter. Ctrl+C to quit.")
print("="*55 + "\n")

db       = JarvisDB(config.DB_PATH)
pipeline = AgentPipeline(db)


def on_status(msg, _color):
    print(f"  [{msg.strip()}]")


def on_sentence(sentence):
    # LLM replies stream in sentence by sentence, exactly as they'd be spoken
    print(f"  >> {sentence}", flush=True)


while True:
    try:
        query = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)

    if not query:
        continue

    print()
    t0 = time.time()

    try:
        response, source, info = pipeline.run(query, on_status=on_status, on_sentence=on_sentence)
        elapsed = time.time() - t0

        source_label = {
            "tool":   "INSTANT (tool)",
            "cache":  "CACHED",
            "memory": "MEMORY",
            "llm":    "LLM (streamed above)",
            "error":  "ERROR",
        }.get(source, source)

        print(f"\nJarvis [{source_label}] ({elapsed:.1f}s):")
        if source != "llm":
            print(f"  {response}")

        if info.get("tool_name") and info["tool_name"] not in ("none", "memory"):
            print(f"  Tool used  : {info['tool_name']}")
        if info.get("cache_score"):
            print(f"  Cache score: {info['cache_score']:.3f}")

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    print()
