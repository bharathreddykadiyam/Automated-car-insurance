import numpy as np
from PIL import Image
import engine

# create uploads dir if missing
import os
os.makedirs('static/uploads', exist_ok=True)

img = np.ones((256,256,3), dtype=np.uint8) * 255
# simulate a damaged area (dark region)
img[60:200, 20:120, :] = [30, 30, 30]
img_path = 'static/uploads/demo_test.jpg'
Image.fromarray(img).save(img_path)

print('Saved demo image to', img_path)
res = engine.engine(img_path)
print('Result:')
print(res)
