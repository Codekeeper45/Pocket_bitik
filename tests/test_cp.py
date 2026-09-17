import sys
import os
import unittest

sys.path.append('/home/hermes/projects/Pocket_bitik')
import bot_new
from telethon.tl.types import (
    MessageMediaPhoto, MessageMediaDocument, MessageMediaPoll,
    MessageMediaWebPage, Document, DocumentAttributeVideo,
    DocumentAttributeAudio, DocumentAttributeSticker, DocumentAttributeAnimated
)

class TestCpCommand(unittest.TestCase):
    def test_parse_tg_message_link(self):
        test_links = [
            ('https://t.me/c/1234567890/456', (-1001234567890, 456, False)),
            ('https://t.me/c/1234567890/10/456', (-1001234567890, 456, False)),
            ('https://t.me/c/1234567890/10/456?single', (-1001234567890, 456, True)),
            ('https://t.me/c/987654321/777?single=1', (-100987654321, 777, True)),
            ('https://t.me/durov/123', ('durov', 123, False)),
            ('https://t.me/durov/123?single', ('durov', 123, True)),
            ('t.me/durov/999', ('durov', 999, False)),
            ('https://telegram.me/telegram/100', ('telegram', 100, False)),
            ('tg://resolve?domain=telegram&post=200', ('telegram', 200, False)),
            ('tg://privatepost?channel=1234567890&post=300', (-1001234567890, 300, False)),
            ('/cp https://t.me/c/12345/67', (-10012345, 67, False)),
            ('текст https://t.me/mychan/555 еще текст', ('mychan', 555, False)),
        ]
        for text, expected in test_links:
            res = bot_new._parse_tg_message_link(text)
            self.assertEqual(res, expected, f"Failed for {text}")

    def test_classify_message_media(self):
        class DummyMsg:
            def __init__(self, media=None, photo=None, video=None, voice=None, audio=None, sticker=None, gif=None, document=None):
                self.media = media
                self.photo = photo
                self.video = video
                self.voice = voice
                self.audio = audio
                self.sticker = sticker
                self.gif = gif
                self.document = document

        m_none = DummyMsg()
        self.assertEqual(bot_new._classify_message_media(m_none), 'none')

        m_web = DummyMsg(media=MessageMediaWebPage(webpage=None))
        self.assertEqual(bot_new._classify_message_media(m_web), 'webpage')

        m_poll = DummyMsg(media=MessageMediaPoll(poll=None, results=None))
        self.assertEqual(bot_new._classify_message_media(m_poll), 'poll')

        m_photo = DummyMsg(media=MessageMediaPhoto(), photo=True)
        self.assertEqual(bot_new._classify_message_media(m_photo), 'photo')

        m_voice = DummyMsg(media=MessageMediaDocument(), voice=True)
        self.assertEqual(bot_new._classify_message_media(m_voice), 'voice')

        m_sticker = DummyMsg(media=MessageMediaDocument(), sticker=True)
        self.assertEqual(bot_new._classify_message_media(m_sticker), 'sticker')

        m_gif = DummyMsg(media=MessageMediaDocument(), gif=True)
        self.assertEqual(bot_new._classify_message_media(m_gif), 'gif')

        doc_video = Document(id=1, access_hash=1, file_reference=b'', date=None, mime_type='video/mp4', size=10, dc_id=1, attributes=[DocumentAttributeVideo(duration=10, w=100, h=100)])
        m_vid = DummyMsg(media=MessageMediaDocument(), video=True, document=doc_video)
        self.assertEqual(bot_new._classify_message_media(m_vid), 'video')

        doc_round = Document(id=1, access_hash=1, file_reference=b'', date=None, mime_type='video/mp4', size=10, dc_id=1, attributes=[DocumentAttributeVideo(duration=10, w=100, h=100, round_message=True)])
        m_round = DummyMsg(media=MessageMediaDocument(), video=True, document=doc_round)
        self.assertEqual(bot_new._classify_message_media(m_round), 'roundvideo')

        doc_file = Document(id=1, access_hash=1, file_reference=b'', date=None, mime_type='application/zip', size=10, dc_id=1, attributes=[])
        m_doc = DummyMsg(media=MessageMediaDocument(), document=doc_file)
        self.assertEqual(bot_new._classify_message_media(m_doc), 'doc')

    def test_check_media_permission(self):
        full_allow = {k: True for k in ['media', 'photos', 'videos', 'roundvideos', 'audios', 'voices', 'docs', 'stickers', 'gifs', 'polls']}
        self.assertTrue(bot_new._check_media_permission('photo', full_allow))
        self.assertTrue(bot_new._check_media_permission('video', full_allow))
        self.assertTrue(bot_new._check_media_permission('doc', full_allow))

        ban_media = full_allow.copy()
        ban_media['media'] = False
        self.assertFalse(bot_new._check_media_permission('photo', ban_media))
        self.assertFalse(bot_new._check_media_permission('video', ban_media))
        self.assertFalse(bot_new._check_media_permission('doc', ban_media))
        self.assertTrue(bot_new._check_media_permission('none', ban_media))
        self.assertTrue(bot_new._check_media_permission('webpage', ban_media))

        ban_docs = full_allow.copy()
        ban_docs['docs'] = False
        self.assertFalse(bot_new._check_media_permission('doc', ban_docs))
        self.assertTrue(bot_new._check_media_permission('photo', ban_docs))

        ban_photos = full_allow.copy()
        ban_photos['photos'] = False
        self.assertFalse(bot_new._check_media_permission('photo', ban_photos))
        self.assertTrue(bot_new._check_media_permission('video', ban_photos))

    def test_help_sections(self):
        self.assertIn('cp', bot_new._HELP_SECTIONS)
        index_text = bot_new._help_index('test')
        self.assertIn('`cp`', index_text)

if __name__ == '__main__':
    unittest.main()
