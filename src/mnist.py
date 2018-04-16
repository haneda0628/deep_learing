import os.path
import gzip
import numpy as np
import pickle
from PIL import Image

key_file = {
  'train_img':'train-images-idx3-ubyte.gz',
  'train_label':'train-labels-idx1-ubyte.gz',
  'test_img':'t10k-images-idx3-ubyte.gz',
  'test_label':'t10k-labels-idx1-ubyte.gz'
}
dataset_dir = os.path.dirname(os.path.abspath(__file__))
img_size = 28 * 28

def load_mnist(filename):
  file_path = dataset_dir + '/' + filename
  with gzip.open(file_path, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
  return data.reshape(-1, img_size)

def load_label(filename):
  file_path = dataset_dir + '/' + filename
  with gzip.open(file_path, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=8)
  return data

label = load_label(key_file['train_label'])
print label[5]
data = load_mnist(key_file['train_img'])
img1 = data[5].reshape(28, 28)
pil_img = Image.fromarray(np.uint8(img1))
pil_img.show()
