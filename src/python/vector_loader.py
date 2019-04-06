# vector_loader.py
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

import numpy as np


class Loader(object):
    """ A class to load the data (word vectors) and to prepare it for the
        training. """

    def __init__(self, data_files, s_shaped=None):
        self.train_in_file = data_files[0]
        self.train_out_file = data_files[1]
        if len(data_files) > 2:
            self.test_in_file = data_files[2]
            self.test_out_file = data_files[3]
        else:
            self.test_in_file = None
            self.test_out_file = None
        self.s_shaped = s_shaped

    def get_vector_size(self):
        """ Returns the size of the word vectors. """
        with open(self.train_in_file, 'r') as vector_file:
            vector_size = vector_file.readline().count(" ")
        return (vector_size+1)

    def get_output_size(self):
        """ Returns the number of classification options found in the data. """
        with open(self.train_out_file, 'r') as output_file:
            num_of_output_neurons = output_file.readline().count(" ")+1
        return num_of_output_neurons
        
    def load_data(self):
        """ Loads the word vectors. """
        print("Loading data...")
        with open(self.train_in_file, 'r') as train_in:
            train_vectors = np.loadtxt(train_in)
        with open(self.train_out_file, 'r') as train_out:
            train_output = np.loadtxt(train_out)

        if self.test_in_file and self.test_out_file:
            with open(self.test_in_file, 'r') as test_in:
                test_vectors = np.loadtxt(test_in)
            with open(self.test_out_file, 'r') as test_out:
                test_output = np.loadtxt(test_out)
        else:
            test_vectors = None
            test_output = None

        if self.s_shaped is True or self.s_shaped is False:
            # I.e. "s_shaped" is not "None" and therefore the data will be
            # prepared for a CNN.
            train_vectors, test_vectors = self.reshape_data(train_vectors,
                                                            test_vectors)

        print("- Done.")
        return train_vectors, train_output, test_vectors, test_output

    def reshape_data(self, train_vectors, test_vectors):
        """ Reshapes the data (one-dimensional vectors) so that a CNN can work
            with it. """
        print("- Done.\nReshaping data...")
        vector_size = len(train_vectors[0])
        primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
                  61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
                  131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
                  193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257,
                  263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331,
                  337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401,
                  409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467,
                  479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563,
                  569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631,
                  641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709,
                  719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797,
                  809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877,
                  881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967,
                  971, 977, 983, 991, 997]
        x = 0
        for prime in primes:
            if (prime > vector_size):
                break
            if (prime == vector_size):
                x += 1
                break
        first_dimension = self.divisable_only_by_primes(primes, (vector_size+x))
        if first_dimension == 0:
            first_dimension = int(np.sqrt(vector_size+x))
        second_dimension = int((vector_size+x)/first_dimension)
        
        reshaped_train = np.empty([len(train_vectors), first_dimension,
                                   second_dimension, 1])
        for i in range(0, len(train_vectors)):
            y = 0
            for j in range(0, first_dimension):
                if self.s_shaped is True:
                    if (j%2) == 0:
                        for k in range(0, second_dimension):
                            reshaped_train[i][j][k][0] = train_vectors[i][y]
                            y += 1
                    else:
                        for k in range((second_dimension-1), 0, -1):
                            reshaped_train[i][j][k][0] = train_vectors[i][y]
                            y += 1
                else:
                    for k in range(0, second_dimension):
                        reshaped_train[i][j][k][0] = train_vectors[i][y]
                        y += 1
        if test_vectors is not None:
            reshaped_test = np.empty([len(test_vectors), first_dimension,
                                      second_dimension, 1])
            for i in range(0, len(test_vectors)):
                y = 0
                for j in range(0, first_dimension):
                    if self.s_shaped is True:
                        if (j%2) == 0:
                            for k in range(0, second_dimension):
                                reshaped_test[i][j][k][0] = test_vectors[i][y]
                                y += 1
                        else:
                            for k in range((second_dimension-1), 0, -1):
                                reshaped_test[i][j][k][0] = test_vectors[i][y]
                                y += 1
                    else:
                        for k in range(0, second_dimension):
                            reshaped_test[i][j][k][0] = test_vectors[i][y]
                            y += 1
                        
        return reshaped_train, reshaped_test

    def divisable_only_by_primes(self, primes, vector_size):
        for prime0 in primes:
            if prime0 > (vector_size/2):
                return 0
            if (vector_size%prime0 == 0):
                for prime1 in primes:
                    if (vector_size/prime0) == prime1:
                        return prime0
                return 0
        return 0
