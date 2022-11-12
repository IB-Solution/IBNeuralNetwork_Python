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

nbEpoch = int(input("nbEpoch maximal :"))

axisEpoch = []
axisCPUTime = []
axisGPUTime = []

brain: ConvolutionalNeuralNetwork = ConvolutionalNeuralNetwork(4, 1, LAYER_OUTPUT, FUNCTION_SIGMOID)
brain.AddLayer(8, LAYER_HIDDEN, FUNCTION_SIGMOID)
brain.SaveNetwork("testNetwork.json")


print("Start GPU")
brain.LoadNetwork("testNetwork.json")
brain.Train(dataset, expected, nbEpoch, 0.25, True, False)
# brain.Train(dataset, expected, nbEpoch, 0.25, False, False)
print("Done GPU")

# Taux de reussite
print("Taux de reussite :")
for x in range(0, len(dataset)):
    print(brain.Predict(dataset[x]), expected[x])
