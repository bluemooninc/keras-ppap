import os
import glob
from PIL import Image

class resize:
  def from_to(self,from_f,to_f,size):
    files = glob.glob(from_f+'*')
    for f in files:
      img = Image.open(f)
      img_resize = img.resize((size, size))
      new_f = to_f + os.path.basename(f)
      print new_f
      img_resize.save(new_f)

if __name__ == '__main__':
  dataset = resize()
  dataset.from_to('./data/org/','./data/150/',150)
