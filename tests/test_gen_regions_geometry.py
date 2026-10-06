from PIL import Image
from gen_regions_geometry import bbox_to_pixels, merge_overlapping_boxes, feather_mask, paste_repair, validate_bbox, plan_regions, blend_patch


def test_geometry_and_compositing():
    assert bbox_to_pixels((.2, .25, .6, .75), (100, 80), .1) == (10, 12, 70, 68)
    assert bbox_to_pixels((0, 0, .1, .1), (10, 10), .5) == (0, 0, 6, 6)
    assert validate_bbox([0, 0, 1, 1]) == [0., 0., 1., 1.]
    for invalid in ([0,0,0,1], [-.1,0,.2,1], [0,0,float('nan'),1], None):
        assert validate_bbox(invalid) is None
    assert merge_overlapping_boxes([(0,0,4,4),(3,2,7,5),(6,4,9,8),(20,0,22,2)]) == [(0,0,9,8),(20,0,22,2)]
    assert merge_overlapping_boxes([(0,0,2,2),(2,0,4,2)], touching=False) == [(0,0,2,2),(2,0,4,2)]
    mask = feather_mask((15,13),4)
    assert all(mask.getpixel((x,0)) == 0 and mask.getpixel((x,12)) == 0 for x in range(15))
    assert all(mask.getpixel((0,y)) == 0 and mask.getpixel((14,y)) == 0 for y in range(13))
    assert mask.getpixel((7,6)) > 0
    src = Image.new('RGB',(12,10),(10,20,30)); patch = Image.new('RGB',(6,4),(200,100,50))
    out = paste_repair(src,patch,(3,2,9,6))
    assert out.getpixel((0,0)) == src.getpixel((0,0)) and out.getpixel((4,3)) == (200,100,50)
    blend = blend_patch(src,patch,(3,2,9,6))
    assert blend.getpixel((0,0)) == src.getpixel((0,0))
    assert blend.getpixel((3,2)) == src.getpixel((3,2))
    findings=[{'bbox':[.1,.1,.3,.3]}, {'bbox':[.25,.25,.4,.4]}, {'bbox':[.8,.8,.9,.9]}, {'bbox':[2,0,3,1]}]
    planned=plan_regions(findings,(100,100),margin=0)
    assert len(planned)==2 and len(planned[0]['findings'])==2

if __name__ == '__main__':
    test_geometry_and_compositing()
    print('PASS: geometry, validation, merge, feather, paste/blend, region planning')
