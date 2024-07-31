import tensorflow as tf
import tf2onnx
import onnxruntime as ort
import numpy as np


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


data_id = str(np.load("data/data_id.npy"))
X = np.load('data/x_sim_'+data_id+'.npy')
# Step 1: Convert TensorFlow model to ONNX
model_path = 'output/fret'+data_id+'.keras'
onnx_path = 'onnx/model'+data_id+'.onnx'


# Load your TensorFlow model
model = tf.keras.models.load_model(model_path)

# Convert the model
spec = (tf.TensorSpec((None, None, 3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)
print(f"Model saved to {onnx_path}")