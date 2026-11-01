"""Text chat in your terminal (no Gnani needed) - quickest way to test the brain:  python chat.py"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.agent import SupportAgent
from src.kb import load_chunks
from src.retriever import Retriever

if __name__ == "__main__":
    retriever = Retriever(load_chunks())
    print(f"[retrieval: {retriever.mode}]  type 'quit' to exit, 'new' for a new conversation\n")
    agent = SupportAgent(retriever)
    while True:
        text = input("You: ").strip()
        if text.lower() in ("quit", "exit"):
            break
        if text.lower() == "new":
            agent = SupportAgent(retriever)
            print("--- new conversation ---")
            continue
        r = agent.respond(text)
        print(f"Agent [{r.decision}]: {r.reply}")
        print(f"   sources: {', '.join(s['id'] for s in r.sources)} | tools: {[c['name'] for c in r.tool_calls]}\n")
