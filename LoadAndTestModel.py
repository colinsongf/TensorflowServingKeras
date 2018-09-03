import tensorflow as tf
from tensorflow import keras
from gen_captcha import gen

tf.enable_eager_execution()


def toText(vec):
  return tf.argmax(tf.reshape(vec, [-1, 4, 10]), 2)


def main():
  model = keras.models.load_model('captcha_model.h5')

  x_test, y_test = next(gen(10))
  results = model.predict(x_test)

  for index, result in enumerate(results):
    print('predict:', toText(result).numpy(), '; real:', toText(y_test[index]).numpy())


if __name__ == '__main__':
  main()
