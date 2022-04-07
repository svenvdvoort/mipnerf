from unprocess import unprocess
from process import process
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import rawpy

tf.config.experimental_run_functions_eagerly(True)
tf.enable_eager_execution()

def read_jpg(filename):
  """Reads an 8-bit JPG file from disk and normalizes to [0, 1]."""
  image_file = tf.read_file(filename)
  # image = tf.image.decode_jpeg(image_file, channels=3)
  image = tf.image.decode_jpeg(image_file, channels=3)
  white_level = 255.0
  # with tf.Session() as sess:
  #   f, img = sess.run([image_file, image])
  #   print(f)
  return tf.cast(image, tf.float32) / white_level

images = glob.glob('D:/ComputerScience/DL/NERF/mipnerf/lego_small/lego/*/*')
new_dir = 'D:/ComputerScience/DL/NERF/mipnerf/lego_small_raw/'


for path in images:
  im = read_jpg(path)
  im, meta = unprocess(im)
  new_path = new_dir + path.split('/')[6]
  # rawpy.imread(new)
  with open(new_path, 'wb') as f:
    np.save(f, np.array(im))
  # cv2.imwrite(new_path, np.array(im))

# for path in images:
#   im = cv2.imread(path)
#   im = cv2.resize(im, (50,50))
#   new_path = new_dir + path.split('/')[6]
#   cv2.imwrite(new_path, im)
#   # rawpy.imread(new)
  # with open(new_path, 'wb') as f:
  #   np.save(f, np.array(im))



# im = process(im, meta['red_gain'], meta['blue_gain'], meta['cam2rgb'])
# # image =
# plt.imshow(im)
# plt.show()
# img = np.array(im, np.uint8)
# img = np.delete(img, 2, 2)
#
# # img = cv2.cvtColor(img, cv2.COLOR_BayerBGGR2RGB)
# cv2.demosaicing(img)
# cv2.imshow('yep', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

