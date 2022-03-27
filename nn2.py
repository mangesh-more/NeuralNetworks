import numpy as np

# X = (hours sleeping, hours studying)
X = np.array(([3,5],[5,1],[10,2]))
# y = Score on test
y = np.array((75,82,93))

# normalize
X= X/np.amax(X,axis=0)
y = y/100

class NeuralNetwork(object):
    def __init__(self):
        # Hyperparameters
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        #forward propagation
        self.Z2 = np.dot(X,self.W1)
        self.a2 = self.sigmoid(self.Z2)
        self.Z3 = np.dot(self.a2,self.W2)
        yHat = self.sigmoid(self.Z3)
        return yHat
        
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

NN = NeuralNetwork()
yHat = NN.forward(X)
yHat

