import os
import numpy as np
from PIL import Image

pathname="D:\\Python3.8.2\\MyPython\\VOC_bird\\SegmentationClass"
savepath="D:\\Python3.8.2\\MyPython\\VOC_bird\\Mask"
filename_list=os.listdir(pathname)

for name in filename_list:
    fullname=pathname+"\\"+name
    Mimg=np.array(Image.open(fullname))
    height,width=Mimg.shape
    for x in range(height):
        for y in range(width):
             if(Mimg[x,y]==3):
                   Mimg[x,y]=255
             else:
                   Mimg[x,y]=0
    newimg=Image.fromarray(Mimg)
    newimg.save(savepath+"\\"+name,"PNG")
