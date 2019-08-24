import tensorflow as tf
from tensorflow.python.saved_model import utils, signature_def_utils, signature_constants
import os


def convert_model_to_tf(trained_checkpoint_prefix, version):
  export_dir = os.path.join('models', str(version))
  loaded_graph = tf.Graph()
  with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)

    serialized_tf_yolo = tf.placeholder(tf.string, name='tf_yolo')
    feature_configs = {'x': tf.FixedLenFeature(shape=[1, 3, 416, 416], dtype=tf.float32), }
    tf_yolo = tf.parse_example(serialized_tf_yolo, feature_configs)
    x = tf_yolo['x']

    tensor_info_x = utils.build_tensor_info(x)
    tensor_info_y = utils.build_tensor_info(tf.placeholder(tf.float32, shape=(None, 8)))

    prediction_signature = signature_def_utils.build_signature_def(
      inputs={'image': tensor_info_x},
      outputs={'scores': tensor_info_y},
      method_name=signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={
                                           'predict_image': prediction_signature,
                                         },
                                         legacy_init_op=legacy_init_op)
    builder.save()
    # print("Successfully saved as TensorFlow Model!")
