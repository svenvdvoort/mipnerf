from unprocess import unprocess
from unprocess import random_ccm
from unprocess import random_gains
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
tf.set_random_seed(4200)


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


images = glob.glob('D:/ComputerScience/DL/NERF/mipnerf/lego_200x200/lego/*/*')
new_dir = 'D:/ComputerScience/DL/NERF/mipnerf/lego_200x200_raw/'

# to unprocess the rgb images to raw
for path in images:
  im = read_jpg(path)
  im, meta = unprocess(im)
  new_path = new_dir + path.split('/')[6]
  # rawpy.imread(new)
  with open(new_path, 'wb') as f:
    np.save(f, np.array(im))
  # cv2.imwrite(new_path, np.array(im))


# images = glob.glob('D:/ComputerScience/DL/NERF/mipnerf/nerf_synthetic/lego/*/*')
# new_dir = 'D:/ComputerScience/DL/NERF/mipnerf/lego_200x200/'
# # to resize the images
# for path in images:
#   im = cv2.imread(path)
#   im = cv2.resize(im, (200,200))
#   new_path = new_dir + path.split('/')[6]
#   cv2.imwrite(new_path, im)



# images = glob.glob('D:/ComputerScience/DL/raw_im/test_preds/*')
# new_dir = 'D:/ComputerScience/DL/rgb_im/'
#
# #to process the raw images to rgb again
# rgb2cam = random_ccm()
# cam2rgb = tf.matrix_inverse(rgb2cam)
#
# rgb_gain, red_gain, blue_gain = random_gains()
#
# for path in images:
#   im = np.load(path)
#   im = tf.convert_to_tensor(np.array([im]))
#   im = tf.cast(im, tf.float32) / 255.0
#   processed_im = np.array(process(im, red_gain, blue_gain, cam2rgb))
#   new_path = new_dir + path.split('\\')[1].split('.')[0] + '.png'
#   # processed_im_rgb = 255 * cv2.cvtColor(processed_im[0,:,:,:], cv2.COLOR_GRB)
#   processed_im_rgb = 255 * processed_im[0,:,:,:][:,:, [1,0,2]]
#   im_norm = cv2.normalize(processed_im_rgb,None, 0, 255,cv2.NORM_MINMAX)
#   cv2.imwrite(new_path, im_norm)


# im = tf.convert_to_tensor(np.array([np.load('D:/ComputerScience/DL/NERF/mipnerf/lego_small_raw/lego/test/r_0.png')]))
# im = tf.cast(im, tf.float32)#/ white_level
# processed_im = np.array(process(im, red_gain, blue_gain, cam2rgb))
# cv2.imwrite("test.png", 255 * processed_im[0,:,:,:])
# cv2.imshow('yep', processed_im[0,:,:,:])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# to process the images
# im = process(im, meta['red_gain'], meta['blue_gain'], meta['cam2rgb'])
# # image =
# plt.imshow(im)
# plt.show()
# img = np.array(im, np.uint8)
# img = np.delete(img, 2, 2)
#
# # img = cv2.cvtColor(img, cv2.COLOR_BayerBGGR2RGB)
# cv2.demosaicing(img)


