# **NEURAL NETWORKS STYLE TRANSFER**

**Neural Style Transfer Algorithms:**
- Neural Style Transfer is the task of changing the style of an image in one domain to the style of an image in another domain. Neural Style Transfer or NST refers to a class of software algorithms that manipulate digital images or videos in order to adopt the appearance or visual style of another image. NST algorithms are characterized by their use of deep neural networks for the sake of image transformation.

**Libraries and Dependencies:**
- I have listed all the necessary Libraries and Dependencies required for this Project here:
```javascript
from d2l import torch as d2l
from IPython import display
import matplotlib.pyplot as plt

import os
import torch                                     
import torchvision
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
```

**Preprocessing and Postprocessing:**
- I will initialize the composite image as a content image. The content image is the only variable that needs to be updated in the style transfer process. Now, I will read the content and style images. I will inspect the coordinate axes of images. I will define the functions for image preprocessing and postprocessing. The preprocess function normalizes each of the three RGB channels of the input images and transforms the results to a format that can be input to the CNN. The postprocess function restores the pixel values in the output image to their original values before normalization.
![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20154.PNG)

**Extracting Features:**
- I will use VGG-19 model pretrained on ImageNet dataset to extract image features. I will select the outputs of different layers from the VGG Neural Network for matching local and global styles. I will define the loss functions which are used for style transfer. The loss functions include content loss, style loss and total variation loss. I have presented the implementation of Function for Extracting Features and Square Error Loss Function using PyTorch here in the Snapshot. 
![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20155.PNG)

**Creating and Initializing Composite Images:**
- In Style Transfer, the composite image is the only variable that needs to be updated. So, I will define a simple model in which composite image is treated as a model parameter. 
![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20156a.PNG)

**Training the Model:**
- During model training, I will constantly extract the content and style features of composite image and calculate the loss function.
![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20156b.PNG)
