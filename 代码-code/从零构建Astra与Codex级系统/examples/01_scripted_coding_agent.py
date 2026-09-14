"""Exercise a real repository-edit loop without requiring trained weights."""

import tempfile
from pathlib import Path

from astra_codex.agent import ScriptedBackend
from astra_codex.coding import build_coding_agent


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "demo.py").write_text("value = 1\n", encoding="utf-8")

        backend = ScriptedBackend(
            [
                '{"tool":"filesystem","arguments":{"action":"read","path":"demo.py"}}',
                '{"tool":"edit","arguments":{"path":"demo.py","old":"value = 1","new":"value = 2"}}',
                '{"tool":"filesystem","arguments":{"action":"read","path":"demo.py"}}',
                "Updated demo.py and re-read the file to verify value = 2.",
            ]
        )
        agent = build_coding_agent(backend, root)
        result = agent.run("Change value from 1 to 2 and verify it.")
        print(result.final_answer)
        print((root / "demo.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
