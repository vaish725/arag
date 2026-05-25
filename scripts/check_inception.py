"""Quick check script: ensure Inception deterministic evidence-only run returns Christopher Nolan with citation 1."""
from arag import agent_loop
import sys


def main():
    q = "Who directed Inception and did they win any Academy Awards?"
    res = agent_loop.run_agent(q, allow_fallback=False)
    ans = res.get("answer", "")
    citations = res.get("citations", []) or []
    print("ANSWER:\n", ans)
    print("CITATIONS:", citations)
    if "Christopher Nolan" not in ans:
        print("TEST FAILED: expected 'Christopher Nolan' in answer")
        sys.exit(2)
    if not citations or citations[0] != 1:
        print("TEST FAILED: expected citation 1 for director, got", citations)
        sys.exit(3)
    print("TEST PASSED")


if __name__ == '__main__':
    main()
