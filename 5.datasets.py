#
# MNIST
# 
dset.MNIST(root, train = TRUE, transform = NONE, target_transform = None, download = FALSE)

# root − root directory of the dataset where processed data exist.
# train − True = Training set, False = Test set
# download − True = downloads the dataset from the internet and puts it in the root.

#
# COCO
#
import torchvision.datasets as dset
import torchvision.transforms as transforms
cap = dset.CocoCaptions(root = '', annFile = 'json annotation file', transform = transforms.ToTensor())
print('Number of samples: ', len(cap))
print(target)

# Output
# Number of samples: 82783
# Image Size: (3L, 427L, 640L)