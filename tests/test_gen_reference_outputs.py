from gen_prompt import parse_intent_plan
from gen_series import series_plan_from_intent


def test_independent_pages_use_only_assigned_sources():
    intent = parse_intent_plan({'output_kind':'independent_images','output_count':2,
        'prompt':'Two separate portraits','scenes':['First portrait','Second portrait'],
        'refs':[{'ref_id':'catalog:1','roles':['subject'],'outputs':[1]},
                {'ref_id':'catalog:2','roles':['subject'],'outputs':[2]},
                {'ref_id':'catalog:3','roles':['style']}]}, 'two portraits')
    assert not intent.fallback_used
    pages = series_plan_from_intent(intent).pages
    assert pages[0].refs == ('catalog:1','catalog:3')
    assert pages[1].refs == ('catalog:2','catalog:3')
    assert 'catalog:2:' not in pages[0].prompt
    assert 'catalog:1:' not in pages[1].prompt
