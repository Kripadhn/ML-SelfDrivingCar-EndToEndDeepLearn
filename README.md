# ML-SelfDrivingCar-EndToEndDeepLearn
Self-driving cars in Machine Learning: End-to-end deep learning
There are several machine learning models that can be used for computer vision based control of self-driving cars. These models typically involve the use of convolutional neural networks (CNNs) and recurrent neural networks (RNNs) to process and analyze images from cameras mounted on the car.

One example of a computer vision based control system for self-driving cars is the end-to-end deep learning model. This model takes raw images from cameras mounted on the car as input and outputs a prediction of the steering angle to control the car. The code for this model might look like this in Python using the TensorFlow library

In this example, the training_images.npy file would contain the raw images from the cameras mounted on the car, while the training_labels.npy file would contain the corresponding steering angles that the model should be trying to predict. The model is trained on this data using the mean squared error loss function and the Adam optimization algorithm, with the final trained model being saved for later use in a file named model.h5.
