import os
import tensorflow as tf
from utils import random_rep_data_gen, set_tf_memory_growth
from model import get_model

data_gen = random_rep_data_gen()

model = get_model(0.1, 0.1, 0.1, 0.1)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.representative_dataset = data_gen.generator
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_rep = converter.convert()
tflite_model_path = "./tf/model.tflite"
open(tflite_model_path, "wb").write(tflite_rep)
tf.lite.experimental.Analyzer.analyze(model_content=tflite_rep)
model.summary()


#os.system("vela --output-dir {} --config ./velaL.ini --system-config MinskyL --memory-mode MinskyL --cpu-tensor-alignment 128 {}".format('./tf', tflite_model_path))
os.system("vela --output-dir {} --config vela.ini --accelerator-config ethos-u55-32  --system-config Minsky --memory-mode Minsky --cpu-tensor-alignment 128 {}".format('./tf', tflite_model_path))
