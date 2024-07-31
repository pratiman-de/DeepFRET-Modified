# Deep-FRET [Modified] #

This is a modified version of [DeepFRET-Model package](https://github.com/hatzakislab/DeepFRET-Model). 

[Paper](https://elifesciences.org/articles/60404)

## Modifications ##
1. The codes are modified to Keras 2.x version and notebooks are available to test a quick run and optimization.
2. It only contains the LSTM model and I've added GRU (works 2x as fast with almost same performance) as well.
3. The current version of pomegranate package gives error. The package versions are also listed.
4. The code also converts the keras model to onnx so that it can be run on any devices easily.
5. We use iSMS software to generate txt files for traces. The `fret_prediction.ipynb` notebook gives a nice visualiztion of predcition (from onnx model and txt traces) and raw data.
6. The parameters are optimized for our data.

