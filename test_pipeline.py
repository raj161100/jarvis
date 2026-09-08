"""
Simulated pipeline test — runs without the GUI.
Type queries and see the full pipeline response in the terminal.
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
        response, source, parsed = pipeline.run(query, on_status=on_status)
        elapsed = time.time() - t0

        source_label = {
            "tool":    "INSTANT (tool)",
            "cache":   "CACHED",
            "blocked": "BLOCKED",
            "llm":     "LLM",
        }.get(source, source)

        print(f"\nJarvis [{source_label}] ({elapsed:.1f}s):")
        print(f"  {response}")

        if parsed.get("tool_name") and parsed["tool_name"] != "none":
            print(f"  Tool used : {parsed['tool_name']}")
        if parsed.get("cache_score"):
            print(f"  Cache score: {parsed['cache_score']:.3f}")
        if parsed.get("complexity"):
            print(f"  Complexity : {parsed['complexity']}")

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    print()
