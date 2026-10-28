import ast
from pathlib import Path
from creative_boundaries import CREATIVE_BOUNDARIES


def test_adult_non_graphic_creativity_and_limits():
    for part in ('совершеннолетними', 'по взаимному согласию', 'поцелуи', 'за кадр', 'Не описывай половые акты', 'не сексуализируй несовершеннолетних'):
        assert part in CREATIVE_BOUNDARIES


def test_shared_prompt_integration():
    root = Path(__file__).resolve().parents[1]
    for name, targets in [('bot_new.py', ['ASK_SYSTEM_PROMPT', 'AUTO_REPLY_SYSTEM_PROMPT']), ('voice_agent_tools.py', ['VOICE_HINT'])]:
        tree = ast.parse((root / name).read_text())
        linked = {n.target.id for n in tree.body if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and 'CREATIVE_BOUNDARIES' in ast.unparse(n.value)}
        assert set(targets) <= linked
