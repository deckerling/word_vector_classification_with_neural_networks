// neural_network.h

// Copyright 2019 E. Decker
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef WORD_VECTOR_CLASSIFICATION_WITH_NEURAL_NETWORKS_SRC_CPP_NEURAL_NETWORK_H_
#define WORD_VECTOR_CLASSIFICATION_WITH_NEURAL_NETWORKS_SRC_CPP_NEURAL_NETWORK_H_

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

class NeuralNetwork {
 public:
  // Constructor for a network to be trained:
  NeuralNetwork(const std::vector<std::vector<unsigned>>& train_output,
      const std::vector<unsigned>& test_output,
      const std::vector<std::vector<std::string>>& words,
      const unsigned epochs,
      const unsigned mini_batch_size,
      const double lambda,
      const double learning_rate,
      const unsigned activation_function_in_fully_connected_layers,
      const std::string& save_file,
      const std::string& load_file);
  // Constructor for a pre-trained network that should do its duty:
  NeuralNetwork(const unsigned input_size,
      const std::vector<std::vector<std::string>>& words,
      const std::string& save_directory,
      const std::string& load_file);

 protected:
  // Data:
  const std::vector<std::vector<unsigned>> train_output_;
  const std::vector<unsigned> test_output_;
  const unsigned train_size_, test_size_;
  const std::vector<std::vector<std::string>> words_;
  // Hyper-parameters:
  const unsigned epochs_, mini_batch_size_;
  const double lambda_, learning_rate_;
  const unsigned mini_batch_size_remainder_, activation_function_in_fully_connected_layers_;
  const std::string save_file_, load_file_;

  std::vector<std::vector<double>> biases_, nabla_b_;
  std::vector<std::vector<std::vector<double>>> weights_, nabla_w_;

  static unsigned GetRandomInt(const unsigned index) { // random generator for
                                                       // shuffling mini_batches
    return (std::rand()%index);
  }

  // Activation functions and their derivatives:
  double ActivationFunctionInFullyConnectedLayers(const double netinput) {
    switch (activation_function_in_fully_connected_layers_) {
     case 0:
      return ReLU(netinput);
     case 1:
      return LeakyReLU(netinput);
     case 2:
      return Sigmoid(netinput);
     default:
      return tanh(netinput);
    }
  }
  double DerivatedActivationFunctionInFullyConnectedLayers(const double netinput) {
    switch (activation_function_in_fully_connected_layers_) {
     case 0:
      return DerivatedReLU(netinput);
     case 1:
      return DerivatedLeakyReLU(netinput);
     case 2:
      return DerivatedSigmoid(netinput);
     default:
      return DerivatedTanh(netinput);
    }
  }
  double ReLU(const double netinput) {
    return (netinput > 0)? netinput : 0;
  }
  double LeakyReLU(const double netinput) {
    return (netinput > 0)? netinput : (0.01*netinput);
  }
  double Sigmoid(const double netinput) {
    return (1/(1+std::exp(-netinput)));
  }
  double DerivatedReLU(const double netinput) {
    return (netinput > 0)? 1 : 0;
  }
  double DerivatedLeakyReLU(const double netinput) {
    return (netinput > 0)? 1 : 0.01;
  }
  double DerivatedSigmoid(const double netinput) {
    double sig = Sigmoid(netinput);
    return (sig*(1-sig));
  }
  double DerivatedTanh(const double netinput) {
    double th = tanh(netinput);
    return (1-th*th);
  }

  void SaveClassifications(std::vector<std::vector<std::string>>& classified_words);

 private:
  unsigned CheckMiniBatchSize(const unsigned mini_batch_size) {
    if (mini_batch_size > train_size_) {
      std::cout << "WARNING: The selected mini batch size is greater than the training size. Thus, it will be set equal to the training size." << std::endl;
      return train_size_;
    }
    return mini_batch_size;
  }
  unsigned GetActivationFunctionOfFullyConnectedLayers(const std::string& load_file, const unsigned activation_function_in_fully_connected_layers);
};

#endif // WORD_VECTOR_CLASSIFICATION_WITH_NEURAL_NETWORKS_SRC_CPP_NEURAL_NETWORK_H_
