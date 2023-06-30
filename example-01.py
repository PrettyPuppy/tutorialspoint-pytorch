import torch
import torch.nn as nn

# define input size, hidden layer size, output size and batch size respectically
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

# create dummy input and target tensors (data)
x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

# create a model
model = nn.Sequential(nn.Linear(n_in, n_h),
                      nn.ReLU(),
                      nn.Linear(n_h, n_out),
                      nn.Sigmoid())

# construct the loss function
criterion = torch.nn.MSELoss()

# construct the optimizer ( stochastic gradient descent in this case )
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# gradient descent
for epoch in range(5000):

    # forward pass: compute predicted y by passing x to the model
    y_pred = model(x)

    # computing and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch, 'loss: ', loss.item())

    # zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad()

    # perform a backward pass (backpropagation)
    loss.backward()

    # update the parameters
    optimizer.step()

