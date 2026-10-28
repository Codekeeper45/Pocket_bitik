import asyncio
import subprocess
from voice_metadata import voice_duration

def test_duration_longer_than_three_minutes():
    # Fast silent fixture reproduces missing duration on long BytesIO OGG upload.
    ogg=subprocess.check_output(['ffmpeg','-v','error','-f','lavfi','-i','anullsrc=r=48000:cl=mono','-t','181.2','-c:a','libopus','-f','ogg','pipe:1'])
    assert asyncio.run(voice_duration(ogg))==182
