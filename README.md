# Deep-FRET [Modified] #

This is a modified version of [DeepFRET-Model package](https://github.com/hatzakislab/DeepFRET-Model). 

[Paper](https://elifesciences.org/articles/60404)

## Modifications ##
1. The codes are modified to Tensorflow 2.x version, and notebooks are available to test a quick run and optimization.
2. It only contains the LSTM model, and I've added GRU (which works 2x as fast with almost the same performance) as well.
3. The current version of the pomegranate package gives an error. The package versions are also listed.
4. The code also converts the Keras model to one so that it can be run on any device easily.
5. We use iSMS software to generate txt files for traces. The `fret_prediction.ipynb` notebook gives a nice visualization of prediction (from the onnx model and txt traces) and raw data.
6. The parameters are optimized for our data.

