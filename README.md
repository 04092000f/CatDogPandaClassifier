# <img src = "https://opencv.org/wp-content/uploads/2021/06/OpenCV_logo_black_.png">
# Implement a CNN-Based Image Classifier

In this project, we have implemented a <code>3-class</code> <b>Image Classifier</b>, using a <b>CNN</b>, on a dataset, which consists of <code>1000</code> images each of <b>panda, cat and dog</b<. We took the dataset from <a href="https://www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda">Kaggle</a>, and split it into <code>80:20</code> ratio for <code>train:validation</code>. To download, <a href="https://www.dropbox.com/sh/n5nya3g3airlub6/AACi7vaUjdTA0t2j_iKWgp4Ra?dl=1">click here</a>. Targetted accuracy on <b>validation data</b> was greater than or equal to <code>85%</code>.
 
## This Project had Two-folded Purpose:
<ul>
<li>Building and training a neural network from scratch in order to understand how to build and train a real network on real data. everything had been implemented this time - right from getting the data to analysing the final results.</li>
<li>Reinforce the use of the complete training pipeline as suggested by OpenCV Univeristy.</li>
</ul>

Refer these two notebooks:
<ol>
<li><a href="https://github.com/04092000f/Image-Classifier-from-Scratch/blob/main/Data_and_Pipeline_Check.ipynb">Data_and_Pipeline_Check.ipynb</a></li>

<b>This notebook is mostly for our own learning</b> and some code is provided by OpenCV Univeristy which goes over the first few steps of data exploration and checking the code in your training pipeline by training a simple model on a sample data. Common sources of errors which you should check are -
<ul>
<li>input and output shape for network.</li>
<li>incompatible shapes among subsequent layers.</li>
<li>loss function and optimizer related.</li>
<li>any image transforms.</li>
<li>plotting of curves.</li>
</ul>
<br>
<li><a href="https://github.com/04092000f/Image-Classifier-from-Scratch/blob/main/Training_from_scratch.ipynb">Training_from_scratch.ipynb</a></li>

<b>Final notebook</b>. We have trained a network on the full data. Experiment with different network parameters to achieve >85% validation accuracy. We might not get 85% on the first attempt. To achieve that, we have experimented with:
<ul>
<li>Number of layers</li>
<li>Parameters inside the layers</li>
<li>Optimizers and learning rate schedulers [You can even get good results without a learning rate scheduler]</li>
<li>Regularization techniques like Data augmentation, Dropout, BatchNorm</li>
<li>Number of epochs</li>
</ul>
</ol>

## Model architecture

### 1. **Input Layer:**
   - **Input**: `(Batch Size, 3, 224, 224)`  
     The input to the model is an image of size `224x224` with 3 channels (RGB). Each image is passed in a batch.

---

### 2. **Convolutional Layers & Pooling:**

   #### First Block:
   - **Conv2dNormActivation**: `(in_channels=3, out_channels=32, kernel_size=7, padding='same')`
     - Applies 32 filters of size 7x7, keeping the spatial resolution the same (`padding='same'`).
     - **Output**: `(Batch Size, 32, 224, 224)`  
   
   - **Conv2dNormActivation**: `(in_channels=32, out_channels=64, kernel_size=5, padding='same')`
     - Applies 64 filters of size 5x5.
     - **Output**: `(Batch Size, 64, 216, 216)`  
   
   - **MaxPool2d**: `(kernel_size=2)`
     - Reduces the spatial dimensions by half using max pooling.
     - **Output**: `(Batch Size, 64, 108, 108)`  

   #### Second Block:
   - **Conv2dNormActivation**: `(in_channels=64, out_channels=128, kernel_size=3, padding='same')`
     - Applies 128 filters of size 3x3.
     - **Output**: `(Batch Size, 128, 108, 108)`  

   - **Conv2dNormActivation**: `(in_channels=128, out_channels=256, kernel_size=3, padding='same')`
     - Applies 256 filters of size 3x3.
     - **Output**: `(Batch Size, 256, 108, 108)`  

   - **MaxPool2d**: `(kernel_size=2)`
     - Reduces the spatial dimensions by half.
     - **Output**: `(Batch Size, 256, 54, 54)`  

   #### Third Block:
   - **Conv2dNormActivation**: `(in_channels=256, out_channels=256, kernel_size=5, padding='same')`
     - Applies 256 filters of size 5x5.
     - **Output**: `(Batch Size, 256, 54, 54)`  

   - **Conv2dNormActivation**: `(in_channels=256, out_channels=512, kernel_size=3, padding='same')`
     - Applies 512 filters of size 3x3.
     - **Output**: `(Batch Size, 512, 52, 52)`  

   - **MaxPool2d**: `(kernel_size=2)`
     - Reduces the spatial dimensions by half.
     - **Output**: `(Batch Size, 512, 26, 26)`  

   #### Fourth Block:
   - **Conv2dNormActivation**: `(in_channels=512, out_channels=256, kernel_size=3, padding='same')`
     - Applies 256 filters of size 3x3.
     - **Output**: `(Batch Size, 256, 26, 26)`  

   - **Conv2dNormActivation**: `(in_channels=256, out_channels=256, kernel_size=3, padding='same')`
     - Applies 256 filters of size 3x3.
     - **Output**: `(Batch Size, 256, 26, 26)`  

   - **Dropout2d**: `(p=0.3)`
     - Drops entire feature maps with a probability of 30% during training for regularization.
     - **Output**: `(Batch Size, 256, 26, 26)`  

   - **MaxPool2d**: `(kernel_size=2)`
     - Reduces the spatial dimensions by half.
     - **Output**: `(Batch Size, 256, 13, 13)`  

---

### 3. **Fully Connected Layers:**

   - **AdaptiveAvgPool2d**: `(output_size=(3, 3))`
     - Applies global average pooling, reducing the spatial dimensions to 3x3, making the output size fixed regardless of input size.
     - **Output**: `(Batch Size, 256, 3, 3)`  

   - **Flatten**
     - Flattens the output from `(Batch Size, 256, 3, 3)` into a 1D vector.
     - **Output**: `(Batch Size, 2304)`  (since `256 * 9 = 2304`)
   
   - **Linear**: `(in_features=2304, out_features=2048)`
     - A fully connected layer that reduces the features to 2048.
     - **Output**: `(Batch Size, 2048)`  

   - **ReLU**: `(inplace=True)`
     - Applies ReLU activation to introduce non-linearity.
     - **Output**: `(Batch Size, 2048)`  

   - **Dropout**: `(p=0.4)`
     - A dropout layer with a 40% probability to prevent overfitting.
     - **Output**: `(Batch Size, 2048)`  

   - **Linear**: `(in_features=2048, out_features=512)`
     - A fully connected layer that reduces the features to 512.
     - **Output**: `(Batch Size, 512)`  

   - **ReLU**: `(inplace=True)`
     - Applies ReLU activation again.
     - **Output**: `(Batch Size, 512)`  

   - **Linear**: `(in_features=512, out_features=3)`
     - The final fully connected layer that produces the 3 class scores for classification.
     - **Output**: `(Batch Size, 3)`  

---

### 4. **Output Layer:**
   - **Output**: `(Batch Size, 3)`  
     The model outputs a vector of size 3 corresponding to the class probabilities or scores for classification.

---

## Summary:

The model architecture is a deep convolutional neural network with:
1. **Convolutional Layers**: For feature extraction.
2. **MaxPooling Layers**: To reduce the spatial dimensions.
3. **Dropout Layers**: For regularization to avoid overfitting.
4. **Fully Connected Layers**: For classification, with ReLU activations and dropout for regularization.


## Final Results

### Accuracy
![Accuracy](https://github.com/04092000f/Image-Classifier-from-Scratch/blob/main/visuals/final_result.png)

From the plots, it can be observed that final <b>validation accuracy</b> is <code>0.9133</code> i.e. <code>91.33%</code>. It means that model is generalizing well on <b>validation data</b> and target <b>validation accuracy</b> of <code>85%</code> is achieved while using the current model architecture.<br>

### Confusion Matrix
![Confusion Matrix](https://github.com/04092000f/Image-Classifier-from-Scratch/blob/main/visuals/confusion_matrix.png)

### Inference
![Inference](https://github.com/04092000f/Image-Classifier-from-Scratch/blob/main/visuals/sample_predictions.png)
