import os
from PIL import Image
imgpath="D:\\2020_courses\\CVDL\\project\\UNETbird\\Image"
maskpath="D:\\2020_courses\\CVDL\\project\\UNETbird\\Mask"
imgsave="D:\\2020_courses\\CVDL\\project\\UNETbird\\train\\Image"
masksave="D:\\2020_courses\\CVDL\\project\\UNETbird\\train\\Mask"

imglist=os.listdir(imgpath)
masklist=os.listdir(maskpath)

for i in range(len(imglist)):
    if(i%50==0):
        print(i)
    img=Image.open(imgpath+"\\"+imglist[i])
    img=img.resize((350,350),Image.ANTIALIAS)
    name=os.path.splitext(imglist[i])[0]
    img.save(imgsave+"\\"+name+".png")
    mask=Image.open(maskpath+"\\"+masklist[i])
    mask=mask.resize((350,350),Image.ANTIALIAS)
    name=os.path.splitext(masklist[i])[0]
    mask.save(masksave+"\\"+name+".png")
    
