# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup-images/bar-chart-before.png "Bar chart of data"
[image2]: ./writeup-images/original.png "Original image"
[image3]: ./writeup-images/grayed.png "After grayscale"
[image4]: ./writeup-images/augmented.png "Augmented data"
[image5]: ./writeup-images/augmented-chart.png "Augmented data chart"
[image6]: ./writeup-images/4.png "Traffic Sign 1"
[image7]: ./writeup-images/5.png "Traffic Sign 2"
[image8]: ./writeup-images/6.png "Traffic Sign 3"
[image9]: ./writeup-images/7.png "Traffic Sign 4"
[image10]: ./writeup-images/8.png "Traffic Sign 5"
[image11]: ./writeup-images/results.png "Traffic Sign 5"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


You're reading it! and here is a link to my [project code](https://github.com/Gryton/carnd-traffic-signs/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy array properties and numpy library "unique" function to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing labels distribution among training data.
From this chart I see, that there are big differences between amount of samples for different labels. I think that such
distribution can favour labels that have many samples, so they will have higher probability of being taken as a
classified result. I'd like to avoid such bahaviour, so I think that I need to flatten this chart a bit.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale, simply because I couldn't get working net with colour
images. I thought that neural network should work better with colour images, but probably I'm not as experianced to 
create one that's working well. So, I converted images to grayscale as I found them simpler to manage. I normalized the data
using some custom normalization to get rid of 0's and 1's, as I was affraid that they can make unexpected local minimas.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image3]

As a last step, I normalized the image data because neural network needs normalized input data to learn.

I decided to generate additional data because of low accuracy of network (something about 80% - I don't know how could
I achieve 89% with no modifications of LeNet network, it didn't work for me).

To add more data to the the data set, I put few lines of code that will add only data that has weak (which for me means
under average) representation in data set. I'm adding additional images by rotating randomly choiced image from class
that's under represented, until it'll reach average count of samples (I calculate average only once, so it's not quite
average number of samples, as I'm adding new once, but it's enough).

Here is an example of an augmented image:

![alt text][image4]

The difference between the original data set and the augmented data set is seen on bar char - now it's "flatten", so
I expect that no labels would be favoured.

![alt text][image5]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is mostly based on lab LeNet network. I introduced more depth in convolutional layers, as I find that's
something that really matters, and improves accuracy. I changed max pooling to average, and that also helped me a lot.
Then I added dropout after fully connected layers, and then I haven't found much better accuracy, but I think that
network learns better (I don't see overlearning, which I see before), and accuracy is more stable - I don't see much
differences between trainings, and accuracy on test set is also better this way.
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride,  outputs 14x14x12		     		|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32	|
| Flatten               | outputs 800                                   |
| Fully connected		| outputs 200           						|
| RELU					|												|
| Dropout               |                                               |
| Fully connected		| outputs 84               						|
| RELU					|												|
| Dropout               |                                               |
| Fully connected		| outputs 43              						|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used exactly the same number of epochs and batch size as in LeNet lab, I also used the same Adam optimizer.
I modified learning rate to higher one, but I obtained overlearning network, than I reduced but hadn't any improves, so I stayed with 0.001, the same as in lab.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.945
* test set accuracy of 0.925


Firstly I took LeNet lab network and tried with RGB images. I couldn't get over 85%, so I decided to change to grayscale
images. During first tries I learned, that adding depth to convolutional layers is a very good way to improve accuracy,
but this model wasn't enough for RGB images. Sometimes I saw real loss in accuracy between epochs, network wasn't training smoothly.

After changing to grayscale images I started from the same starting point, with bare LeNet lab network. Than I added depth,
but it wasn't enough - I had something like 91%. I've read a bit about augmenting data, and that was something that changed everything - then
I realised that data is very unbalanced in classes, and it's necessary to add more balance. I received network that has ability
to go over 93%, but sometimes I saw regression in accuracy. I've found that dropout can add more stability to my network,
so I've added 2 dropouts after fully connected layers, and received pretty stable network, with ability to show
nice accuracy also on test set. 
I also tried adding one more convolutional layer, but it made big regression for my model - probably it was also
wrong point to do this, as I had not bad functioning model, maybe I should starting tuning for whole model after adding
this layer - so I went back. I tried also changing learning rate, but I found that my starting point was really good enough.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I've downloaded from Slack, as one of students uploaded additional images:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

Image quality of this images is really good. Last one can be difficult to classify, as it's shot from angle, and also
"forbidden over 3.5t" graphic is similar to other graphics. Rest images shouldn't be a problem, maybe there can be small
problem with speed limits, as generally these signs are similar to each other - there's only few pixels difference in first
digit, but I hope everything should work.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h         		| 30 km/h   									| 
| 70 km/h     			| 70 km/h 										|
| Children crossing     | Children crossing								|
| Ahead only    		| Ahead only					 				|
| Over 3.5t prohibited	| Over 3.5t prohibited 							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100% (even 8 of 8 as I tested additional 8 images). 
I think that's because of good quality of images, and is a bit higher than test set, but the "test batch" is really small - it's hard to prognose taking only these images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is not so sure that this is a speed limit 70 km/h (probability of 0.44), but the image does contain a 70 km/h speed limit. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .44         			| 70 km/h        								| 
| .17     				| 20 km/h                                   	|
| .16					| 100 km/h						            	|
| .15	      			| 120 km/h     					 				|
| .15				    | 30 km/h                           			|


For the second image situation is similar, but network is even less sure (only 0.17), but it does see the difference to
the other images - so image of speed limit 30 km/h is also recognised well. As I thought, speed limits are similar to each
other, so they can be misleaded - but it seems, that network can notice details enough.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .17         			| 30 km/h        								| 
| .11     				| 20 km/h                                   	|
| .9					| 70 km/h						            	|
| .8	      			| 50 km/h     					 				|
| .6				    | end of limit 80 km/h                 			|

Third image also hasn't big probability of classification, but next one is 2 times less - so Children crossing is also
recognised right. Probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .26         			| Children crossing        								| 
| .14     				| Beware of ice/snow                                  	|
| .11					| Slippery road						            	|
| .9	      			| Bicycles crossing     					 				|
| .9				    | Right-of-way at the next intersection                			|

Fourth image is recognised with more certainity (0.36). Interesting is, that Ahead only could be mislead with
Yield (second place). I wonder how network see them similar, it shows it perceives signs completely different than
human. Probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .36         			| Ahead only      								| 
| .20     				| Yield                                     	|
| .14					| Go straight or left						            	|
| .11	      			| Go straight or right   					 				|
| .9				    | No passing for vehicles over 3.5 metric tons                			|

I thought that fifth image will be difficult, but it seems that net was most certain on this one (from five mentioned here).
It seems that icon of truck is so characteristic that net created good pattern for it.
Probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .49         			| Vehicles over 3.5 metric tons prohibited      | 
| .19     				| Roundabout mandatory                            |
| .17					| No passing for vehicles over 3.5 metric tons  |
| .16	      			| End of passing     					 		|
| .15				    | End of no passing by vehicles over 3.5 metric tons |

Image below contains all 8 signs I tested as new signs.
![alt text][image11]