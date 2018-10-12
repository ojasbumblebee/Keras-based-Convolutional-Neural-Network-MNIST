'''Trains a simple convnet on the MNIST dataset.

Training done on MX150 GPU
'''

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.utils import plot_model
from keras import optimizers

#batch_size = 32
num_classes = 10
epochs = 5

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#reshpaing input images
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')

#input training data is split into 50,000 training and 10,000 validation set x_train_train=x_train[0:50000]
x_train_val=x_train[50000:60000]

#10,000 testing examples
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#splitting  training and validation
y_train_train=y_train[0:50000]
y_train_val=y_train[50000:60000]


#hyper-parameters for grid search
learning_rates = [0.1, 0.01, 0.001, 0.0001]
batch_sizes = [32, 64, 128]


#Logging output for each combination of hyper parameters
def log_output_for_current_hyperparameter_combination(output_log):
    with open("output_log_file.txt", 'a') as f:
        f.write(output_log)

#Main training
count_1 = 1
count_2 = 100
for learning_rate in learning_rates:
    for batch_size in batch_sizes: 

        #defining model - 2 convolutional layers followed by 1 fully connected layer with a droput probabilty of 0.25 
        #Followed by Another FC for output to 10 classes with dropout 0.5 
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(1,28,28), data_format='channels_first'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        #Stochastic Gradient Descent 
        optimizer=optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        history = model.fit(x_train_train, y_train_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_train_val, y_train_val))

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        output_log ='\nFor combination of batch size:'+str(batch_size)+'learning rate:'+str(learning_rate)+'\n'+\
                    'Test loss:' + str(score[0]) +'\n'+'Test accuracy:'+str(score[1]) 

        log_output_for_current_hyperparameter_combination(output_log)
    
        # Plot training & validation accuracy values
        fig1 = plt.figure(count_1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy, batch size: '+str(batch_size)+' learning rate: '+str(learning_rate))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        fig1.savefig('myfig_accuracy_'+str(batch_size)+'_'+str(learning_rate)+'.png')
        #plt.show()

        fig2 = plt.figure(count_2)
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss, batch size: '+str(batch_size)+' learning rate: '+str(learning_rate))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        #plt.show()
        fig2.savefig('myfig_loss_'+str(batch_size)+'_'+str(learning_rate)+'.png')
        
        count_1 += 1
        count_2 += 1

#plot the model diagram
plot_model(model, to_file='model.png')

