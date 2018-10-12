# Keras-based-Convolutional-Neural-Network-MNIST
---

All the training was performed on Nvidia- MX150 GPU

The data set was divided into 50,000 training, 10,000 validation and 10,000 testing images 

The repository contains codes:


cnn_keras_hyperparameter_grid_search_batch_size_&_learning_rate.py - Used to perform a grid search on batch size and learning rate values with number of epochs equal to 5 as constant and activation as Relu(Rectified Linear Unit).

learning rates used - 0.1, 0.01, 0.001, 0.0001, 0.00001
batch sizes used - 32, 64, 128

To run code
```
python cnn_keras_hyperparameter_grid_search_batch_size_&_learning_rate.py
```


cnn_keras_hyperparameter_activations.py - Used to perform a search to find the best activation function as an hyper parameter.  

activation functions used are - Relu, Sigmoid, Tanh

To run code:

```
python cnn_keras_hyperparameter_activations.py

```

cnn_keras_final_model.py - Final model with tuned hyperparameters

```
python cnn_keras_final_model.py
```


For the grid search outputs and different hyperparameter set outputs check [output graphs folder](https://github.ncsu.edu/ovbarve/Keras-based-Convolutional-Neural-Network-MNIST/tree/master/Output_graphs).

