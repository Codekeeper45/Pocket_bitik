import json
import unittest
from PIL import Image
from gen_image_layers import LayerError, parse_plan, composite_layers

class LayerTests(unittest.TestCase):
    def plan(self, **kw):
        p={"style":"ink","lighting":"left key","perspective":"eye-level","background":{"prompt":"room"},"requested_identities":["Ada","Bo","Cy","De"],"layers":[{"prompt":"four distinct people","identities":["Ada","Bo","Cy","De"],"x":0,"y":0,"w":1,"h":1}]}
        p.update(kw); return p
    def test_keeps_more_than_three_names(self):
        p=parse_plan(json.dumps(self.plan()),100,100,"Ada Bo Cy De")
        self.assertEqual(len(p['layers'][0]['identities']),4)
    def test_rejects_missing_name_and_bad_bounds(self):
        p=self.plan(); p['layers'][0]['identities'].pop()
        with self.assertRaises(LayerError): parse_plan(json.dumps(p),100,100,"names")
        p=self.plan(); p['layers'][0]['x']=.8
        with self.assertRaises(LayerError): parse_plan(json.dumps(p),100,100,"names")
    def test_real_alpha_composites_and_opaque_rejected(self):
        bg=Image.new('RGBA',(40,40),'white'); fg=Image.new('RGBA',(10,10),(255,0,0,0))
        for y in range(2,8):
            for x in range(2,8): fg.putpixel((x,y),(255,0,0,255))
        out=composite_layers(bg,[(fg,{"x":.25,"y":.25,"w":.5,"h":.5})],(40,40))
        self.assertEqual(out.getpixel((20,20))[:3],(255,0,0))
        with self.assertRaises(LayerError): composite_layers(bg,[(Image.new('RGBA',(10,10),'red'),{"x":0,"y":0,"w":1,"h":1})],(40,40))
if __name__=='__main__': unittest.main()
