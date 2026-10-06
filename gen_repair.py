"""Context crop preparation and inverse mapping without geometric distortion."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class CropMapping:
    source_box: tuple[int,int,int,int]
    canvas_size: tuple[int,int]
    scale: float
    offset: tuple[int,int]
    content_size: tuple[int,int]
    def source_to_canvas(self, x: float, y: float) -> tuple[float,float]:
        x1,y1,_,_=self.source_box; return ((x-x1)*self.scale+self.offset[0],(y-y1)*self.scale+self.offset[1])
    def canvas_to_source(self, x: float, y: float) -> tuple[float,float]:
        x1,y1,_,_=self.source_box; return ((x-self.offset[0])/self.scale+x1,(y-self.offset[1])/self.scale+y1)
    def response_box_to_source(self, box: Sequence[float]) -> tuple[int,int,int,int]:
        if len(box)!=4: raise ValueError("box must have four coordinates")
        a=self.canvas_to_source(box[0],box[1]); b=self.canvas_to_source(box[2],box[3])
        x1,y1,x2,y2=self.source_box
        return (max(x1,int(round(a[0]))),max(y1,int(round(a[1]))),min(x2,int(round(b[0]))),min(y2,int(round(b[1]))))

def validate_bbox(box, image_size):
    if not isinstance(box,(tuple,list)) or len(box)!=4: raise ValueError("bbox must contain four coordinates")
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in box): raise ValueError("bbox coordinates must be numeric")
    x1,y1,x2,y2=box; w,h=image_size
    if any(not float(v).is_integer() for v in box): raise ValueError("bbox coordinates must be integer pixels")
    if not (0<=x1<x2<=w and 0<=y1<y2<=h): raise ValueError("bbox outside image or empty")
    return tuple(int(v) for v in box)

def context_box(box, image_size, *, context=0.35, kind="default"):
    x1,y1,x2,y2=validate_bbox(box,image_size); w,h=image_size
    factor={"hand":0.55,"face":0.40,"default":context}.get(kind,context)
    mx=max(1,int((x2-x1)*factor)); my=max(1,int((y2-y1)*factor))
    return max(0,x1-mx),max(0,y1-my),min(w,x2+mx),min(h,y2+my)

def prepare_context_crop(image, box, *, canvas_size=(1024,1024), context=0.35, kind="default"):
    """Aspect-preserving letterboxed context image and mapping; never stretches."""
    from PIL import Image
    crop_box=context_box(box,image.size,context=context,kind=kind)
    crop=image.crop(crop_box); cw,ch=crop.size; tw,th=canvas_size
    scale=min(tw/cw,th/ch)
    nw,nh=max(1,round(cw*scale)),max(1,round(ch*scale))
    resized=crop.resize((nw,nh),Image.Resampling.LANCZOS)
    canvas=Image.new(image.mode,(tw,th),0 if "A" in image.mode else (0,0,0))
    ox=(tw-nw)//2; oy=(th-nh)//2
    canvas.paste(resized,(ox,oy),resized if "A" in resized.mode else None)
    return canvas,CropMapping(crop_box,(tw,th),scale,(ox,oy),(nw,nh))

def extract_mapped_region(response, mapping: CropMapping, original_box):
    """Map original repair box into generated context crop and extract it preserving aspect."""
    x1,y1,x2,y2=validate_bbox(original_box,(mapping.source_box[2],mapping.source_box[3]))
    a=mapping.source_to_canvas(x1,y1); b=mapping.source_to_canvas(x2,y2)
    l,t,r,bottom=(int(round(a[0])),int(round(a[1])),int(round(b[0])),int(round(b[1])))
    return response.crop((l,t,r,bottom))

__all__=["CropMapping","validate_bbox","context_box","prepare_context_crop","extract_mapped_region"]
