# coding=utf-8
import tensorflow as tf
from tensorflow import keras
from gen_captcha import gen

IMAGE_HEIGHT = 24
IMAGE_WIDTH = 80
IMAGE_DEPTH = 1
MAX_CAPTCHA = 4
CHAR_SET_LEN = 10


def create_model(input_shape, num_classes):
  inputs = keras.Input(shape=input_shape)

  y = inputs
  y = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, padding='SAME')(y)
  y = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(y)
  y = keras.layers.Dropout(0.05)(y)

  y = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, padding='SAME')(y)
  y = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(y)
  y = keras.layers.Dropout(0.05)(y)

  y = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, padding='SAME')(y)
  y = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(y)
  y = keras.layers.Dropout(0.05)(y)

  y = keras.layers.Flatten()(y)
  y = keras.layers.Dense(1024, activation='relu')(y)
  y = keras.layers.Dropout(0.05)(y)

  r = [keras.layers.Dense(CHAR_SET_LEN, activation='softmax', name='c%d' % (i+1))(y) for i in range(MAX_CAPTCHA)]
  r = keras.layers.concatenate(r, axis=-1)

  model = keras.Model(inputs=inputs, outputs=r)
  model.compile(loss='categorical_crossentropy', optimizer=tf.train.AdamOptimizer(learning_rate=0.001), metrics=['accuracy'])

  model.summary()

  return model


def main():
  model = create_model((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH), MAX_CAPTCHA * CHAR_SET_LEN)
  model.fit_generator(gen(), steps_per_epoch=50, epochs=5, verbose=1)

  model.save('captcha_model.h5')

  x_test, y_test = next(gen(128))
  score = model.evaluate(x_test, y_test, verbose=0)

  print(score)


if __name__ == '__main__':
  main()
