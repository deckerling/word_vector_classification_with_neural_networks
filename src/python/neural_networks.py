# neural_networks.py

""" Implementations of a MLP and a CNN. """

from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.regularizers import l2


class MultilayerPerceptron(object):

    def __init__(self, train_vectors, train_output, test_vectors, test_output,
                 layer_sizes, learning_rate, dropout, batch_size, epochs):
        print("Creating a MLP...")
        self.train_in = train_vectors
        self.train_out = train_output
        self.test_in = test_vectors
        self.test_out = test_output
        self.layer_sizes = layer_sizes
        self.sgd = optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0,
                                  nesterov=False)
        self.drop = dropout
        self.batch_size = batch_size
        self.epochs = epochs

        print("- Done.\nTraining started...")
        self.train_network()

    def train_network(self):
        MLP = Sequential()

        if len(self.layer_sizes) > 2:
            MLP.add(layers.Dense(self.layer_sizes[1],
                                 input_dim=self.layer_sizes[0],
                                 activation='tanh'))
            for i in range(2, (len(self.layer_sizes)-1)):
                MLP.add(layers.Dropout(self.drop))
                MLP.add(layers.Dense(self.layer_sizes[i], activation='tanh'))
            MLP.add(layers.Dropout(self.drop))
            MLP.add(layers.Dense(self.layer_sizes[len(self.layer_sizes)-1],
                                 activation='softmax'))
        else:
            MLP.add(layers.Dense(self.layer_sizes[1],
                                 input_dim=self.layer_sizes[0],
                                 activation='sigmoid'))

        MLP.compile(loss='categorical_crossentropy', optimizer=self.sgd,
                    metrics=['accuracy'])
        MLP.fit(self.train_in, self.train_out, batch_size=self.batch_size,
                epochs=self.epochs, verbose=2)

        print("- Training finished.")
        if self.test_in  is not None and self.test_out is not None:
            results = MLP.evaluate(self.test_in, self.test_out, verbose=0)
            print("Final test:\n - Accuracy: "+str(results[1]))

            
class ConvolutionalNeuralNetwork(object):

    def __init__(self, train_vectors, train_output, test_vectors, test_output,
                 num_of_filters, kernel_size, pool_size, learning_rate,
                 regularizer, dropout, batch_size, epochs):
        print("Creating a CNN...")
        self.train_in = train_vectors
        self.train_out = train_output
        self.test_in = test_vectors
        self.test_out = test_output
        self.filters = num_of_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.sgd = optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0,
                                  nesterov=False)
        self.reg = regularizer
        self.drop = dropout
        self.batch_size = batch_size
        self.epochs = epochs

        print("- Done.\nTraining started...")
        self.train_network()

    def train_network(self):
        CNN = Sequential()

        CNN.add(layers.Conv2D(self.filters, self.kernel_size, strides=(1, 1),
                              padding='same', activation='relu',
                              kernel_regularizer=l2(self.reg),
                              input_shape=(len(self.train_in[0]),
                                           len(self.train_in[0][0]), 1)))
        CNN.add(layers.Conv2D(self.filters, self.kernel_size, strides=(1, 1),
                              padding='same', activation='relu',
                              kernel_regularizer=l2(self.reg)))
        CNN.add(layers.MaxPool2D(self.pool_size, padding='same'))
        CNN.add(layers.Dropout(self.drop))
        CNN.add(layers.Flatten())
        CNN.add(layers.Dense(100, activation='tanh',
                             kernel_regularizer=l2(self.reg)))
        CNN.add(layers.Dropout(self.drop))
        CNN.add(layers.Dense(len(self.train_out[0]), activation='softmax',
                             kernel_regularizer=l2(self.reg)))

        CNN.compile(loss='categorical_crossentropy', optimizer=self.sgd,
                    metrics=['accuracy'])
        CNN.fit(self.train_in, self.train_out, batch_size=self.batch_size,
                epochs=self.epochs, verbose=2)

        print("- Training finished.")
        if self.test_in is not None and self.test_out is not None:
            results = CNN.evaluate(self.test_in, self.test_out, verbose=0)
            print("Final test:\n - Accuracy: "+str(results[1]))
