# HandGestureRecognition-using-3D-Conv-and-CNN-RNN-Stack
### Problem Statement
Imagine you are working as a data scientist at a home electronics company which manufactures state of the art smart televisions. You want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote.<br>
The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:

    * Thumbs up:  Increase the volume
    * Thumbs down: Decrease the volume
    * Left swipe: 'Jump' backwards 10 seconds
    * Right swipe: 'Jump' forward 10 seconds  
    * Stop: Pause the movie
The training data consists of a few hundred videos categorised into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames(images). These videos have been recorded by various people performing one of the five gestures in front of a webcam - similar to what the smart TV will use. <br> All images in a particular video subfolder have the same dimensions but different videos may have different dimensions. Specifically, videos have two types of dimensions - either 360x360 or 120x160 (depending on the webcam used to record the videos).
The data can be found at https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL 
 ### Architecture
 
 #### **CNN + RNN architecture**
In this method we pass the images of a video through a CNN which extracts a feature vector for each image, and then pass the sequence of these feature vectors through an RNN.The output of the RNN is a regular softmax (for a classification problem such as this one).
 #### **3D Convolutional Network, or Conv3D**
 3D convolutions are a natural extension to the 2D convolutions. Just like in 2D conv, we move the filter in two directions (x and y), in 3D conv, we move the filter in three directions (x, y and z). In this case, the input to a 3D conv is a video (which is a sequence of 30 RGB images). If we assume that the shape of each image is 100x100x3, for example, the video becomes a 4-D tensor of shape 100x100x3x30 which can be written as (100x100x30)x3 where 3 is the number of channels. Hence, deriving the analogy from 2-D convolutions where a 2-D kernel/filter (a square filter) is represented as (fxf)xc where f is filter size and c is the number of channels, a 3-D kernel/filter (a 'cubic' filter) is represented as (fxfxf)xc (here c = 3 since the input images have three channels). This cubic filter will now '3D-convolve' on each of the three channels of the (100x100x30) tensor.
 
 ### Steps
* Load the training and validation data
* Define the Hyperparameters (batch size and number of epochs)
* Crop and Resize all the images
* Define a custom **Generator** function that will preprocess the images as well as create a batch of video frames.In the Generator a video is represented as (number of images, height, width, number of channels)
* Create a Conv3D model using MaxPooling3D layers and softmax function at the output layer in such a way that the model is able to give good accuracy on the least number of parameters so that it can fit in the memory of the webcam.
* Create a Conv2D+GRU model using TimeDistributed
* Also create a Conv2d+LSTM layer
* Finally create a transfer learning model using MobileNet +LSTM
* Compare the models for training and validation accuracy, number of trainable parameters and training time

### Comparison of the different models is updated in write-up.docx
### Final model is uploaded .

 
