# VPilot
Keras Deep Learning project to build a self-driving agent that takes camera frames as input and predicts steering, throttle and brake commands, trained using the [DeepGTAV](https://github.com/ai-tor/DeepGTAV) self-driving car development environment.

<img src="http://forococheselectricos.com/wp-content/uploads/2016/07/tesla-autopilot-1.jpg" alt="Self-Driving Car" width="800px">

For now Supervised Learning is implemented, but there are plans to improve it using Reinforcement Learning.

## Supervised Learning

The model is an smaller version of AlexNet (influenced by NVIDIA's end-to-end approach) including a layer of LSTM units on top of it to include temporal inference. Training is being done on sequences of contiguous frames (about 5 seconds, 50 frames for 10 Hz rate). See _model.py_ for details.
