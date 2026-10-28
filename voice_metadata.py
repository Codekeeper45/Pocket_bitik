"""Explicit duration metadata for assembled Telegram voice notes."""
import asyncio
import json
import math
import tempfile
from pathlib import Path

async def voice_duration(ogg):
    with tempfile.TemporaryDirectory(prefix='voice-meta-') as tmp:
        path=Path(tmp)/'voice.ogg'
        path.write_bytes(ogg)
        proc=await asyncio.create_subprocess_exec('ffprobe','-v','error','-show_entries','format=duration','-of','json',str(path),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL)
        try:
            raw,_=await asyncio.wait_for(proc.communicate(),30)
        except BaseException:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise
        if proc.returncode:raise ValueError('Cannot inspect voice duration')
        duration=float(json.loads(raw)['format']['duration'])
        if not math.isfinite(duration) or duration<=0:raise ValueError('Invalid voice duration')
        return math.ceil(duration)
