from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# Implementation of simple feedforward network
class FCNet(Model):
  def __init__(self, neurons=[12,12,3], reg=None,activation="relu"):
    super(FCNet, self).__init__()

    self.denseLayers=[]
    for idx,neuron in enumerate(neurons):
      self.denseLayers.append(Dense(neuron, activation="relu"))

    self.outputLayer = Dense(1, activation=None)

  def call(self, input_x):
    output = input_x

    for layer in self.denseLayers:
      output = layer(output)

    return self.outputLayer(output)

 