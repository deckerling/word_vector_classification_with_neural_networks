#!/usr/bin/env python3

# word_vector_classifier.py
#
# Copyright 2019 E. Decker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import neural_networks
import vector_loader

create_CNN = True # if "False", a MLP will be created
# Paths to the files containing the data for the training (and for testing the
# network (optional)); note that the files should only contain values - the
# "words" (i.e. strings) are not supposed to be part of the vector files!:
data_files = ["train_in.txt",
              "train_out.txt",
              "test_in.txt",
              "test_out.txt"]

# Hyper-parameters for both types of network:
learning_rate = 0.1
dropout = 0.5
batch_size = 100
epochs = 20

if create_CNN:
    # Hyper-parameters for a CNN:
    s_shaped_data = True # if "True" the one-dimensional word vectors will be reshaped into two dimensions in s-shape
    loader = vector_loader.Loader(data_files, s_shaped_data)
    num_of_filters = 10
    kernel_size = (3, 3)
    pool_size = (2, 2)
    regularizer = 0.01

    train_vectors, train_output, test_vectors, test_output = loader.load_data()
    neural_networks.ConvolutionalNeuralNetwork(train_vectors, train_output,
                                               test_vectors, test_output,
                                               num_of_filters, kernel_size,
                                               pool_size, learning_rate,
                                               regularizer, dropout, batch_size,
                                               epochs)
else:
    # Hyper-parameters for a MLP:
    loader = vector_loader.Loader(data_files)
    layer_sizes = [loader.get_vector_size(), 100, 30, loader.get_output_size()]
    
    train_vectors, train_output, test_vectors, test_output = loader.load_data()
    neural_networks.MultilayerPerceptron(train_vectors, train_output,
                                         test_vectors, test_output, layer_sizes,
                                         learning_rate, dropout, batch_size,
                                         epochs)
