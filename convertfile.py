import os
import glob
from PIL import Image
filename = '/home/poudelas/Downloads/yo/data/ng/'
i=0
for img in glob.glob(filename + '/*.*'):
    print(img)
    name = os.path.basename(img)
    name = os.path.splitext(name)[0]
    print(name)
    img = Image.open(img)
    new_img = img.resize((1024, 1024),Image.ANTIALIAS)
    new_img.save('/home/poudelas/Downloads/yo/data/ng1/'+str(i)+'.png','png')
    i=i+1

