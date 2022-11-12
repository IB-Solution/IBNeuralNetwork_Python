# IBNeuralNetwork
    Python deeplearning package

# Activation Function
- [x] BINARY STEP Function
- [x] LINEAR Function
- [x] SIGMOID Function
- [x] TANH Function
- [x] RELU Function
- [x] LEAKY RELU Function
- [x] PARAMETERISED RELU Function
- [x] EXPONENTIAL LINEAR UNIT Function

# NeuralNetwork
## Neuron Type
- [ ] Input cell

- [ ] Hidden cell
- [ ] Hidden Recurrent cell
- [ ] Hidden LSTM cell (Long Short Term Memory)

- [ ] Output cell

## Tech
- [ ] CPU calculation
- [ ] GPU calculation
- [ ] Mutation change weight
- [ ] Mutation change biais
- [ ] Mutation add neurons
- [ ] Dropout
- [ ] Save network
- [ ] Load network


# ConvolutionalNeuralNetwork
## Layer Type
- [x] Input layer

- [x] Hidden layer
- [ ] Hidden Recurrent Layer
- [ ] Hidden LSTM layer (Long Short Term Memory)

- [x] Output layer

## Tech
- [x] CPU calculation
- [x] GPU calculation
- [ ] Mutation change weight
- [ ] Mutation change biais
- [ ] Mutation change neurons
- [ ] Mutation add layers
- [ ] Dropout
- [x] Save network
- [x] Load network

## Example
```python
from IBNeuralNetwork.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from IBNeuralNetwork.ConvolutionalNeuralNetwork import LAYER_OUTPUT, LAYER_HIDDEN
from IBNeuralNetwork.ActivationFunction import FUNCTION_SIGMOID

dataset = [
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0],
    [1,1,1,1]
]
expected = [
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1]
]

nbEpoch = int(input("nEpoch :"))

axisEpoch = []
axisCPUTime = []
axisGPUTime = []

brain: ConvolutionalNeuralNetwork = ConvolutionalNeuralNetwork(4, 1, LAYER_OUTPUT, FUNCTION_SIGMOID)
brain.AddLayer(8, LAYER_HIDDEN, FUNCTION_SIGMOID)
brain.SaveNetwork("testNetwork.json")


print("Start GPU")
brain.LoadNetwork("testNetwork.json")
brain.Train(dataset, expected, nbEpoch, 0.25, True, False)
print("Done GPU")

print("Rate :")
for x in range(0, len(dataset)):
    print(brain.Predict(dataset[x]), expected[x])

```


# Genetic Algorithm
