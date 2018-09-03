import tensorflow as tf
from tensorflow import keras as K
import os

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS


def save_model_to_serving(model, export_version, export_path='captcha_servable'):
  print(model.input, model.output)
  signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={'image': model.input}, outputs={'captcha': model.output})
  export_path = os.path.join(tf.compat.as_bytes(export_path), tf.compat.as_bytes(str(export_version)))
  builder = tf.saved_model.builder.SavedModelBuilder(export_path)
  legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
  builder.add_meta_graph_and_variables(
      sess=K.backend.get_session(),
      tags=[tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          'hack_captcha': signature,
      },
      legacy_init_op=legacy_init_op)
  builder.save()


def main():
  model = K.models.load_model('captcha_model.h5')
  save_model_to_serving(model, 2, export_path='captcha_servable')


if __name__ == '__main__':
  main()
