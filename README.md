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




Justification of hyper- parameters:
---

The Network was evaluated for three hyper parameters namely learning rate, batch size and activation functions at hidden layers. A grid search was performed on batch size and learning rate as the two input hyper parameters. We tested for the permutations of the following namely learning rate as 0.1, 0.01, 0.001, 0.0001, 0.00001 and batch size as 32, 64, 128. After running our grid search algorithm we reached the conclusion that the network gave the best outputs for the batch size of 32 and learning rate of 0.01 for Relu activation functions. Next up we tested the network using the previously obtained values of hyper parameters namely learning rate - 0.01 and batch size 32 on different activation functions. We tried 3 different activation functions namely Relu, Sigmoid and Tanh. We obtained the best test results for the Tanh activation function. We conclude that the results obtained on the said hyperparameters were better than other permutations that we experimented as the Tanh activation has a steeper learning rate than Relu and converges faster in our scenario as we used only 5 epochs to train our network and also the learning rate being used i.e. 0.01 is on the higher side. Also due to a smaller size of the network i.e. the network being comparitively shallow we did not observe any problem of exploding gradients with Tanh as an activation function. We can obtain a better performance with Relu as an activation by increasing the number of epochs as it will take more time to converge to optimal values as it introduces sparsity into the model as ca be seen by below output. 

The ouput for 12 epochs ofor Relu activation:
Epoch 12/12
60000/60000 [==============================] - 25s 422us/step - loss: 0.0255 - acc: 0.9913 - val_loss: 0.0268 - val_acc: 0.9906
Test loss: 0.0267825192


Clearly we can see that we get better performance for Relu activation with higher epochs on the same conditions but the increase is marginal at the cost of resources utilised in form of time and number of epochs. Sow we can get a comparable performance with Tanh activation for just 5 epochs without losing much on accuracy as shown below.

Final Log For combination of with activation as tanh and batch size:32 and learning rate:0.01
Epochs 5
Test loss:0.04035723799202824
Test accuracy:0.9872
