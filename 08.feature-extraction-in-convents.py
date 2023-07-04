#
# Step 1 : Import the respective models to create the feature extraction model with PyTorch
#
import torch
import torch.nn as nn
from torchvision import models

#
# Step 2 : Create a class of feature extractor which can be called as and when needed
#
class Feature_extractor(nn.module):
   def forward(self, input):
      self.feature = input.clone()
      return input
   
new_net = nn.Sequential().cuda() # the new network
target_layers = [conv_1, conv_2, conv_4] # layers you want to extract`
i = 1
for layer in list(cnn):
   if isinstance(layer,nn.Conv2d):
      name = "conv_"+str(i)
      art_net.add_module(name,layer)
      if name in target_layers:
         new_net.add_module("extractor_"+str(i),Feature_extractor())
      i+=1
   if isinstance(layer,nn.ReLU):
      name = "relu_"+str(i)
      new_net.add_module(name,layer)
   if isinstance(layer,nn.MaxPool2d):
      name = "pool_"+str(i)
      new_net.add_module(name,layer)
new_net.forward(your_image)
print (new_net.extractor_3.feature)
