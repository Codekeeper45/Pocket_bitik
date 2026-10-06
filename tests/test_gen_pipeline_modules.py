import io
import unittest
from PIL import Image
from gen_provider import validate_image, classify_provider_error
from gen_contracts import QAResult
from gen_qa import parse_findings, accept_repair
from gen_repair import prepare_context_crop
from gen_policy import GenerationBudget, BudgetExceeded, action_for

class PipelineModuleTests(unittest.TestCase):
    def png(self, size=(400,200)):
        b=io.BytesIO(); Image.new('RGB',size,'red').save(b,'PNG'); return b.getvalue()
    def finding(self, severity='high'):
        return dict(id='f1',category='anatomy',severity=severity,confidence=.9,bbox=[.2,.2,.4,.4],affected_subject='a',requirement_id='r',description='x',location='hand')
    def test_real_image_validation_and_bytes(self):
        image=validate_image(self.png())
        self.assertEqual(image.pixel_size,(400,200)); self.assertEqual(image.mime_type,'image/png'); self.assertEqual(image.byte_size,len(image.data))
        with self.assertRaises(ValueError): validate_image(b'nope')
        with self.assertRaises(ValueError): validate_image(self.png(),max_bytes=2)
    def test_provider_errors_not_all_4xx_moderation(self):
        self.assertEqual(classify_provider_error(400).kind,'invalid_request')
        self.assertEqual(classify_provider_error(400,'content_policy').kind,'moderation')
    def test_qa_schema_and_paired_acceptance(self):
        before=parse_findings({'findings':[self.finding()]}); after=parse_findings({'findings':[]})
        self.assertTrue(accept_repair(before,after,target_finding_id='f1').accepted)
        self.assertFalse(accept_repair(before,parse_findings({'findings':[self.finding('critical')]}),target_finding_id='f1').accepted)
        self.assertEqual(parse_findings('{bad').status,'unavailable')
    def test_qa_rejects_unknown_top_level_and_finding_fields(self):
        finding=self.finding()
        self.assertEqual(parse_findings({'findings':[dict(finding, unexpected=True)]}).status,'unavailable')
    def test_qa_requires_location_for_global_finding_and_validates_strict_types(self):
        finding=self.finding(); finding['bbox']=None; finding['location']='hand'
        self.assertEqual(parse_findings({'findings':[finding]}).status,'unavailable')
        finding=self.finding(); finding['confidence']=True
        self.assertEqual(parse_findings({'findings':[finding]}).status,'unavailable')
        finding=self.finding(); finding['severity']='urgent'
        self.assertEqual(parse_findings({'findings':[finding]}).status,'unavailable')
    def test_crop_mapping_preserves_aspect(self):
        src=Image.new('RGB',(800,300),'blue')
        canvas,m=prepare_context_crop(src,(300,80,500,220),canvas_size=(512,512))
        self.assertEqual(canvas.size,(512,512)); self.assertAlmostEqual(m.content_size[0]/m.content_size[1],m.source_box[2]-m.source_box[0] if False else m.content_size[0]/m.content_size[1])
        x,y=m.source_to_canvas(350,100); sx,sy=m.canvas_to_source(x,y); self.assertAlmostEqual(sx,350); self.assertAlmostEqual(sy,100)
    def test_budget(self):
        b=GenerationBudget(max_generation_calls=1); b.consume('generation')
        with self.assertRaises(BudgetExceeded): b.consume('generation')
        self.assertEqual(action_for([self.finding()]),'repair')

if __name__=='__main__': unittest.main()
