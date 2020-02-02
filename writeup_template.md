# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./plots/test_bar.png "Visualization"
[image2]: ./plots/train_bar.png "Visualization1"
[image3]: ./plots/original.png "Visualization2"
[image4]: ./plots/grayscale.png "Grayscaling"
[image5]: ./plots/his_equa.png "HE"
[image6]: ./plots/adaptive.png "CACHE"

[image7]: ./images/1.png "Traffic Sign 1"
[image8]: ./images/2.JPG "Traffic Sign 2"
[image9]: ./images/3.png "Traffic Sign 3"
[image10]: ./images/4.JPG "Traffic Sign 4"
[image11]: ./images/5.png "Traffic Sign 5"
[image12]: ./plots/5_tests.JPG "Traffic SignS "
[image13]: ./plots/softmax_5.png "Top 5 probabilities "


### Data Set Summary & Exploration

#### 1.Basic summary of the data set

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 examples 
* The size of the validation set is 4410 examples 
* The size of test set is 12630 examples 
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2.Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
It is a bar chart showing how looks  the distribution of classes in the training set and test set.

![alt text][image1]
![alt text][image2]

Here is a plot of  traffic signs images:
![alt text][image3]
As we see images don't look good, some images displayed have a low contrast and a bad illumination.These aspects can influence the accuracy of our trained model.


### Data preprocessing

For object recognition there are a several  parameters that need to be taken in consideration for achieving a good accuracy:
- Variable viewpoint
- Variable illumination
- Scale
- Deformation
- Occlusion
- Background clutter
As a first step, I decided to convert the images to grayscale because using color channels didn’t seem to improve the accuracy of model a lot.Conversion of images from color to grayscale  did not resolve the problem with illumination.Here is an example of a traffic sign image after  grayscaling.
![alt text][image4]
To extract some details from regions that are darker or lighter  of the  image there are two methods implemented in skimage module:
-Histogram Equalization (
-Contrast Limited Adaptive Histogram Equalization (CLAHE)
Here is a useful link about these methods :[]( https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html)
Here is an example of a traffic sign image after applying histogram equalization:
![alt text][image5]
Here is an example of a traffic sign image after applying Contrast Limited Adaptive Histogram Equalization:
![alt text][image6]
Looks good !!

### Design and Test a Model Architecture
#### 1 . Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
I use the LeNet arhitecture.(Yan LeCun)
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1            							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 10x10x16  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6  					|
| Flatten 	        	| outputs 400									| 
| Fully connected		| outputs 120									|
| RELU					|												|
| Fully connected		|outputs 84   									|
| RELU					|												|
| Fully connected		|outputs 43  									|
| Softmax				|												|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used  Adam Optimizer and the following hyperparameters:
Learning rate=0.001
Batch size=32
Epochs=14
For each of this parameters I tried different values and finally I chose the best parameters from my point of view .

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.950
* test set accuracy of 0.931
I chose a well known architecture LeNet.For getting  a higher accuracy for the validation set I did additional steps for data processing with I describe above.
* Why did you believe it would be relevant to the traffic sign application?

It's a relevant arhitecture to the traffic sign application as long as we obtain a very good accuracy for the validation set and test set.
With a good  data  preprocessing of images we can  incresse the accuracy of validation set from 0.890 to ~0.950 using LeNet arhitecture.
Another method for increasing the accuracy is to  increase the number of training dataset twice or 4 times.(Data agumentation). 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11]
The images witch I found from Google weren't   difficult for the model to classify because images looks good.None of them have 
low-contrast ,colors fading, graffiti or other problems that are in the real world.
![alt text][image12]
CACHE transformation on the new images.
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
The performance of the model on the test set was 0.931 and the performance of the new test set (new 5 images) was 1.00.
Which is a good score.


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.10%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the latest 3 cells of the Ipython notebook Traffic_Sign_Classifier-Copy2.ipynb

For the first image, the model is  absolute sure that this is a priority road sign (probability of 1.0), and the image does contain a priority road sign. 
For the third image , the model is relatively sure that this is 0.8 probability a speed limit 80km/h road sign and 0.2 probability think that is a  speed limit 30 km/h , and the image does contain a speed limit 80km/h sign.
For each of the five new images, I create a graphic visualization of the soft-max probabilities.
The top five soft max probabilities were:

![alt text][image13]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


