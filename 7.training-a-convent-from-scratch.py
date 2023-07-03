#
# Step 1 : Create a necessary class with respective parameters.
# The parameters include weights with random value.
#
class Neural_Network(nn.Module):
    def __init__(self,):
        super(Neural_Network, self).__init__()
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)  # 3 * 2 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 3 * 1 tensor

#
# Step 2 : Create a feed forward pattern of function with sigmoid functions.
#
def forward(self, X):
   self.z = torch.matmul(X, self.W1) # 3 X 3 '.dot' 
   self.z2 = self.sigmoid(self.z) # activation function
   self.z3 = torch.matmul(self.z2, self.W2)
   o = self.sigmoid(self.z3) # final activation 
   function
   return o
   def sigmoid(self, s):
      return 1 / (1 + torch.exp(-s))
   def sigmoidPrime(self, s):
      # derivative of sigmoid
      return s * (1 - s)
   def backward(self, X, y, o):
      self.o_error = y - o # error in output
      self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
      self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
      self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
      self.W1 += torch.matmul(torch.t(X), self.z2_delta)
      self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)

#
# Step 3 : Create a training and prediction model as mentioned below −
#
def train(self, X, y):
   # forward + backward pass for training
   o = self.forward(X)
   self.backward(X, y, o)
def saveWeights(self, model):
   # Implement PyTorch internal storage functions
   torch.save(model, "NN")
   # you can reload model with all the weights and so forth with:
   # torch.load("NN")
def predict(self):
   print ("Predicted data based on trained weights: ")
   print ("Input (scaled): \n" + str(xPredicted))
   print ("Output: \n" + str(self.forward(xPredicted)))