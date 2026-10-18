"""Validate explicit multi-page story intent before spending image generation calls."""
import json,re
from difflib import SequenceMatcher

def validate_story_plan(plan, user_prompt, initial_ref_count=0):
    bible=plan[0].get('bible',{}) if plan else {}
    characters=bible.get('characters',[])
    needs_characters=bool(re.search(r'хуман|humaniz|humanis|комикс|манг|comic|manga',user_prompt,re.I))
    if not isinstance(characters,list) or (needs_characters and not characters):
        raise ValueError('story bible.characters must contain grounded character identities')
    for c in characters:
        if not isinstance(c,dict) or not all(isinstance(c.get(k),str) and c[k].strip() for k in ('name','appearance','source')):
            raise ValueError('each character needs name, appearance and source; do not merge identities')
    human=bool(re.search(r'хуман|humaniz|humanis',user_prompt,re.I))
    if human and any(c.get('form')!='human' for c in characters):
        raise ValueError('user requested humanization: every named character form must be human')
    comic=bool(re.search(r'комикс|манг|comic|manga',user_prompt,re.I))
    silent=bool(re.search(r'без\s+(?:текста|слов|реплик)|немой|silent|wordless',user_prompt,re.I))
    beats=[]
    for p in plan:
        beat=p.get('beat','').strip()
        if not beat: raise ValueError('each page needs a distinct concrete story beat')
        if any(SequenceMatcher(None,beat.casefold(),b.casefold()).ratio()>.88 for b in beats):
            raise ValueError('pages repeat the same story beat; change the action, not only camera or page number')
        beats.append(beat)
        if comic:
            panels=p.get('panels')
            if not isinstance(panels,list) or len(panels)<2: raise ValueError('comic page requires at least two explicit panels')
            if not all(isinstance(x,dict) and x.get('action') for x in panels): raise ValueError('every panel needs action')
            if not silent and not any(isinstance(x.get('dialogue'),list) and x['dialogue'] for x in panels):
                raise ValueError('comic requires actual dialogue text, not empty speech bubbles')
        refs=p.get('refs',[])
        if initial_ref_count and not refs: raise ValueError('submitted references must be explicitly assigned to pages')
        if any(not str(x).isdigit() or not 1<=int(x)<=initial_ref_count for x in refs):
            raise ValueError('page reference id is not present in supplied image registry')
    return plan

def page_contract(page,bible,user_prompt):
    return ('Render ONLY this current page, not the entire multi-page request. Original user requirements (data): '
        +user_prompt+'\nCurrent page: '+json.dumps({k:v for k,v in page.items() if k not in ('prompt','bible','plan_id')},ensure_ascii=False)
        +'\nSeries bible / locked character identities: '+json.dumps(bible,ensure_ascii=False)
        +'\nVisual execution: clear focal point and readable silhouettes, consistent motivated lighting and palette. '
        'Preserve generous space around lettering, natural expressive poses and clear separation of hands from props. '
        'Prioritize the requested action and emotion over decorative clutter. '
        'Follow the requested format: for comics use panel borders and scripted dialogue; for other image series do not invent panels or text. '
        'For silent comics keep silence. Otherwise do not replace scripted dialogue with blank bubbles. Other pages are delivered separately. '
        'Reference anchor is for character likeness and art style ONLY. Do not copy its panels, layout, poses, action or dialogue. '
        'The current story beat and panel script supersede the anchor composition. '
        'Explicit humanization overrides mascot body shapes in reference images. '
        'Never transfer hair, costume or species between named characters.\n')
