import unittest,tempfile,stat,ast
from pathlib import Path
from setup_wizard import write_env,validate

class Setup(unittest.TestCase):
 def test_private_config_and_no_overwrite(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'.env';write_env({'KEY':'dummy'},p)
   self.assertEqual(stat.S_IMODE(p.stat().st_mode),0o600)
   with self.assertRaises(FileExistsError):write_env({'KEY':'changed'},p)
   self.assertIn('dummy',p.read_text())
 def test_symlink(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'.env';target=Path(d)/'target';target.write_text('keep');p.symlink_to(target)
   with self.assertRaises(ValueError):write_env({'KEY':'dummy'},p,True)
   self.assertEqual(target.read_text(),'keep')
 def test_validation(self):
  v={'api_id':'12345','api_hash':'a'*32,'OPENAI_COMPAT_BASE_URL':'https://example.com/v1','OPENAI_COMPAT_MODEL':'vendor/model','OPENAI_COMPAT_API_KEY':'dummy'};validate(v)
  for bad in ['http://remote.example/v1','https://user:pass@example.com/v1','https://example.com/v1?key=x']:
   with self.assertRaises(ValueError):validate(dict(v,OPENAI_COMPAT_BASE_URL=bad))
  with self.assertRaises(ValueError):validate(dict(v,OPENAI_COMPAT_API_KEY='${SECRET}'))
 def test_actual_registry_uses_remote_model(self):
  source=Path('bot_new.py').read_text();begin=source.index('CUSTOM_MODELS = _model_state');end=source.index('MODEL_TOOLS_SUPPORT =',begin)
  scope={'_model_state':{},'MODEL_REGISTRY':{'custom-openai':('openai_compat','vendor/model','Custom',128000,1.15)},'_compat_model':'vendor/model','_compat_enabled':True,'_compat_vision':True}
  exec(compile(source[begin:end],'registry','exec'),scope)
  self.assertEqual(scope['MODEL_REGISTRY']['custom-openai'][1],'vendor/model');self.assertEqual(scope['ACTIVE_MODEL'],'custom-openai')
if __name__=='__main__':unittest.main()
