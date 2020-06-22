import os
import numpy as np
from PIL import Image

pathname="D:\\2020_courses\\CVDL\\project\\UNETbird\\train\\Image\\"
filename=os.listdir(pathname)

m0,m1,m2,s0,s1,s2=0,0,0,0,0,0
for name in filename:
    img=Image.open(pathname+name)
    M=np.array(img)
    M=M/255
    m0+=M[:,:,0].mean()
    m1+=M[:,:,1].mean()
    m2+=M[:,:,2].mean()
    s0+=M[:,:,0].std()
    s1+=M[:,:,1].std()
    s2+=M[:,:,2].std()

m0=m0/len(filename)
m1=m1/len(filename)
m2=m2/len(filename)
s0=s0/len(filename)
s1=s1/len(filename)
s2=s2/len(filename)

