### An improvement of my model:

| Layer                 |     Description                                | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x3 RGB image                           | 
| Convolution     | 1x1 stride, valid padding, outputs 28x28x6     |
| RELU             |                                             |
| Max pooling    | 2x2 stride, valid padding, outputs 14x14x6     |
| Convolution     | 1x1 stride, valid padding, outputs 10x10x16          |
| RELU             |                                             |
| Max pooling    | 2x2 stride, valid padding, outputs 5x5x6         |
| Flattening        | Outputs 400                                |
| Fully contected    | Outputs 120                                |
| RELU             |                                             |
| Dropout        | Keep probability train: 0.5, validate/test: 1        |
| Fully contected    | Outputs 84                                    |
| RELU             |                                             |
| Fully contected    | Outputs 43 (number of classes)                |

### TensorBoard visualization tool  could be very handy in hyperparameter selection, debugging and visualization . A few tutorials:
[https://www.youtube.com/watch?v=3bownM3L5zM]
[https://www.youtube.com/watch?v=eBbEDRsCmv4]

### Here are a few ways which could be tested to improve on model accuracy:

- Data augmentation 
- L2 Regularization is also a very good way to introduce non-linearities to model. See this StackOverflow question on L2 regularization.
[https://stackoverflow.com/questions/38286717/tensorflow-regularization-with-l2-loss-how-to-apply-to-all-weights-not-just]

- Here is an article on How To Improve Deep Learning Performance.
[https://machinelearningmastery.com/improve-deep-learning-performance/]

- With images of much bigger sizes, generators will become very helpful in avoiding memory overload. 
Check out this Medium post on Python generators to get an idea of how they work.
[https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98]
