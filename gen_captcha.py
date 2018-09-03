# coding:utf-8
from PIL import Image, ImageDraw, ImageFont
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras


font_size = 28
font_folder = os.path.join('.', 'fonts')

IMAGE_HEIGHT = 24
IMAGE_WIDTH = 80
IMAGE_DEPTH = 1
MAX_CAPTCHA = 4
CHAR_SET_LEN = 10


def gen_dataset(batch_size=128):
  return tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32), (tf.TensorShape((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)), tf.TensorShape((batch_size, MAX_CAPTCHA*CHAR_SET_LEN))))


def gen(batch_size=64):
  while True:
    yield gen_next_batch(batch_size)


def convert2gray(img):
  if len(img.shape) > 2:
    gray = np.mean(img, -1)
    return gray
  else:
    return img


def gen_next_batch(batch_size=128):
  batch_x = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], np.float32)
  batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN], np.int8)

  for i in range(batch_size):
    number = random.randint(0, 9999)
    image = gen_captcha("%04d" % number)

    image = convert2gray(image)
    batch_x[i, :] = (image.flatten() / 255).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))

    arr = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN, dtype=np.int8)
    arr[number // 1000 % 10] = 1
    arr[10+number // 100 % 10] = 1
    arr[20+number // 10 % 10] = 1
    arr[30+number % 10] = 1
    batch_y[i, :] = arr

  return batch_x, batch_y


def gen_captcha(text):
  font_file = os.path.join(
      font_folder, random.choice(os.listdir(font_folder)))
  font = ImageFont.truetype(size=font_size, font=font_file)

  image = Image.new(mode='RGB', size=(IMAGE_WIDTH, IMAGE_HEIGHT), color='#FFFFFF')
  draw = ImageDraw.Draw(im=image)

  size = draw.textsize(text, font=font)
  offset = font.getoffset(text)

  draw.text(xy=(0, 0-offset[1] + random.randint(0, IMAGE_HEIGHT-size[1]+offset[1])), text=text, fill='#FF0000', font=font)

  for i in range(20):
    draw.line(xy=[random.randint(0, 97), random.randint(
        0, 24), random.randint(0, 97), random.randint(0, 24)], fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

  return np.asarray(image)


def main():
  # np.set_printoptions(threshold=np.nan)
  x, y = next(gen(1))
  print(x.tolist())
  print(y)
  # print(x.shape)
  # print(y.shape)


if __name__ == '__main__':
  main()
