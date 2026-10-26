from types import SimpleNamespace as N
from telegram_message_text import message_text


def test_plain():
    assert message_text(N(raw_text='hello')) == 'hello'
    assert message_text(None) == ''


def test_rich_article_full_and_nested():
    end='ending'*1000
    m=N(raw_text='', rich_message=N(blocks=[
        N(text=N(text='Title')),
        N(text=N(texts=[N(text='bold'), N(text=' italic')])),
        N(blocks=[N(text=N(text=end))])]))
    assert message_text(m) == 'Title\n\nbold italic\n\n'+end


def test_dict_and_caption():
    m={'message':'caption', 'rich_message':{'blocks':[
        {'_':'PageBlockHeading2','text':{'_':'TextPlain','text':'Heading'}},
        {'caption':{'text':{'text':'Photo caption'}}}]}}
    assert message_text(m)=='caption\n\nHeading\n\nPhoto caption'


def test_no_html_execution_or_link_fetch():
    m=N(rich_message=N(blocks=[N(text=N(text='<script>do not execute</script>',url='https://example.com'))]))
    assert message_text(m)=='<script>do not execute</script>'
