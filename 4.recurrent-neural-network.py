#
# Step 1 : Import the necessary packages for implementing recurrent neural network
#
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import pylab as pl
#
# Step 2 : Set the model hyper parameters with the size of input layer set to 7.
# There will be 6 context neurons and 1 input neuron for creating target sequence
#
dtype = torch.FloatTensor
input_size, hidden_size, output_size = 7, 6, 1
epochs = 300
seq_length = 20
lr = 0.1
data_time_steps = np.linspace(2, 10, seq_length + 1)
data = np.sin(data_time_steps)
data.resize((seq_length + 1, 1))

x = Variable(torch.Tensor(data[:-1]).type(dtype), requires_grad=False)
y = Variable(torch.Tensor(data[1:]).type(dtype), requires_grad=False)

#
# Step 3 : Weights are initialized in the recurrent neural network using normal distribution with zero mean.
# W1 will represent acceptance of input variables and W2 will represent the output which is generated.
#
w1 = torch.Tensor(input_size, hidden_size).type(dtype)
init.normal(w1, 0.0, 0.4)
w1 = Variable(w1, requires_grad=True)

w2 = torch.Tensor(hidden_size, output_size).type(dtype)
init.normal(w2, 0.3, 0.0)
w2 = Variable(w2, requires_grad=True)

#
# Step 4 : Create a function for feed forward which uniquely defines the neural network
#
def forward(input, context_state, w1, w2):
    xh = torch.cat((input, context_state), 1)
    context_state = torch.tanh(xh.mm(w1))
    out = context_state.mm(w2)
    return (out, context_state)

#
# Step 5: Start training procedure of recurrent neural network's sine wave implementation.
# The outer loop iterates over each loop and the inner loop iterates through the element of sequence.
# here, we will also compute Mean Square Error which helps in the prediction of continuous variables
#
for i in range(epochs):
    total_loss = 0
    context_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad=True)
    for j in range(x.size(0)):
        input = x[j:(j + 1)]
        target = y[j:(j + 1)]
        (pred, context_state) = forward(input, context_state, w1, w2)
        loss = (pred - target).pow(2).sum() / 2
        total_loss += loss
        loss.backward()
        w1.data -= lr * w1.grad.data
        w2.data -= lr * w2.grad.data
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        context_state = Variable(context_state.data)
    if i % 10 == 0:
        print('Epoch: {0} loss {1}'.format(i, total_loss.data))

context_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad=False)
predictions = []

for i in range(x.size(0)):
    input = x[i: i + 1]
    (pred, context_state) = forward(input, context_state, w1, w2)
    context_state = context_state
    predictions.append(pred.data.numpy().ravel()[0])

#
# Step 6 : Plot the sine wave as the way it is needed
#
pl.scatter(data_time_steps[:-1], x.data.numpy(), s = 90, label = 'Actual')
pl.scatter(data_time_steps[1:], predictions)
pl.legend()
pl.show()