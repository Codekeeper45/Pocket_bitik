from voice_agent_tools import VOICE_HINT

def test_scene_defaults_and_explicit_override_are_in_shared_prompt():
    for phrase in ['сценку голосом', '2-4 минуты', '2-4 персонажами', 'send_voice_scene', 'fish_voice_search', 'fish_voice_get', 'одним голосом', 'важнее этого умолчания', 'Не меняй постоянный голос чата']:
        assert phrase in VOICE_HINT

def test_plain_tts_is_not_forced_into_scene():
    assert 'Обычная просьба ответить голосом или озвучить текст не требует нескольких ролей' in VOICE_HINT
