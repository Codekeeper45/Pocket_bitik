import copy,unittest
from gen_story_contract import validate_story_plan,page_contract

def plan():
 bible={'characters':[{'name':'Alice','form':'human','appearance':'purple hair','source':'user'}]}
 return [{'page':1,'beat':'Meets friend at station','panels':[{'action':'Arrives','dialogue':['Привет!']},{'action':'Offers ticket','dialogue':[]}],'refs':['1'],'bible':bible}, {'page':2,'beat':'Finds lost letter in cafe','panels':[{'action':'Opens envelope','dialogue':['Это тебе.']},{'action':'Reads letter','dialogue':[]}],'refs':['1'],'bible':bible}]
class Tests(unittest.TestCase):
 def test_good(self):self.assertEqual(len(validate_story_plan(plan(),'комикс в хуманизации',1)),2)
 def test_empty_bible(self):
  p=plan();p[0]['bible']={}
  with self.assertRaises(ValueError):validate_story_plan(p,'комикс',1)
 def test_mascot(self):
  p=plan();p[0]['bible']['characters'][0]['form']='mascot'
  with self.assertRaisesRegex(ValueError,'humanization'):validate_story_plan(p,'хуманизация',1)
 def test_duplicate_beat(self):
  p=plan();p[1]['beat']=p[0]['beat']
  with self.assertRaisesRegex(ValueError,'repeat'):validate_story_plan(p,'комикс',1)
 def test_blank_dialogue(self):
  p=plan();p[0]['panels'][0]['dialogue']=[]
  with self.assertRaisesRegex(ValueError,'dialogue'):validate_story_plan(p,'комикс',1)
 def test_lost_refs(self):
  p=plan();p[0]['refs']=[]
  with self.assertRaisesRegex(ValueError,'references'):validate_story_plan(p,'комикс',1)
 def test_exact_request_and_page_local(self):
  t=page_contract(plan()[0],plan()[0]['bible'],'Сделай 5 страниц в хуманизации')
  self.assertIn('Сделай 5 страниц в хуманизации',t);self.assertIn('ONLY this current page',t);self.assertIn('Do not copy',t)
if __name__=='__main__':unittest.main()
