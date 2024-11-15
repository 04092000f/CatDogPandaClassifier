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
