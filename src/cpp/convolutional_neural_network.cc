// convolutional_neural_network.cc

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

#include <algorithm>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>

#include "cnn.h"

ConvolutionalNetwork::ConvolutionalNetwork(const std::vector<std::vector<std::vector<double>>>& train_input, const std::vector<std::vector<unsigned>>& train_output, const std::vector<std::vector<std::vector<double>>>& test_input, const std::vector<unsigned>& test_output, const std::vector<std::vector<std::string>>& words, const std::vector<unsigned>& fully_connected_layer_sizes, const unsigned num_of_feature_maps, const unsigned num_of_convolutional_and_pooling_layers, const std::vector<unsigned>& kernel_sizes, const unsigned pooling, const bool leaky_ReLU_in_convolutional_layers, const unsigned activation_function_in_fully_connected_layers, const unsigned epochs, const unsigned mini_batch_size, const double lambda, const double learning_rate, const std::string& save_file, const std::string& load_file, const bool s_shaped_vectors)
    : NeuralNetwork(train_output, test_output, words, epochs, mini_batch_size, lambda, learning_rate, activation_function_in_fully_connected_layers, save_file, load_file),
      train_input_(train_input),
      test_input_(test_input),
      fully_connected_layer_sizes_(fully_connected_layer_sizes),
      num_of_feature_maps_(num_of_feature_maps),
      num_of_convolutional_and_pooling_layers_(num_of_convolutional_and_pooling_layers), // exclusive input layer
      num_of_fully_connected_layers_(fully_connected_layer_sizes.size()), // inclusive output layer
      kernel_sizes_(kernel_sizes),
      even_kernel_sizes_(((((kernel_sizes[0]-1)/2)%2)==0), ((((kernel_sizes[1]-1)/2)%2)==0)), // will be needed in the pooling function
      pooling_(pooling), // 0 = max pooling, 1 = L2 pooling, 2 = average pooling
      leaky_ReLU_in_convolutional_layers_(leaky_ReLU_in_convolutional_layers) {
  std::cout << "- Done." << '\n' << "Initializing network..." << std::endl;
  convolutional_layers_.resize(1+2*num_of_convolutional_and_pooling_layers_);
  convolutional_layers_[0].resize(1);
  convolutional_layers_[0][0] = std::vector<std::vector<double>>(train_input_[0].size(), std::vector<double>(train_input_[0][0].size()));
  for (unsigned i = 0; i < num_of_convolutional_and_pooling_layers_; ++i) {
    convolutional_layers_[2*i+1].resize(num_of_feature_maps_);
    convolutional_layers_[2*i+2].resize(num_of_feature_maps_);
    for (unsigned j = 0; j < num_of_feature_maps_; ++j) {
      convolutional_layers_[2*i+1][j] = convolutional_layers_[2*i][0]; // convolutional layer
      convolutional_layers_[2*i+2][j] = std::vector<std::vector<double>>((((convolutional_layers_[2*i+1][0].size()%2) == 0)? convolutional_layers_[2*i+1][0].size()/2+1 : (convolutional_layers_[2*i+1][0].size()+1)/2+1), std::vector<double>((((convolutional_layers_[2*i+1][0][0].size()%2) == 0)? convolutional_layers_[2*i+1][0][0].size()/2+1 : (convolutional_layers_[2*i+1][0][0].size()+1)/2+1))); // pooling layer
    }
  }
  convolutional_netinputs_.resize(num_of_convolutional_and_pooling_layers_);
  for (unsigned i = 0; i < num_of_convolutional_and_pooling_layers_; ++i)
    convolutional_netinputs_[i] = convolutional_layers_[2*i+1];
  fully_connected_layers_.resize(num_of_fully_connected_layers_);
  for (unsigned i = 0; i < num_of_fully_connected_layers_; ++i)
    fully_connected_layers_[i].resize(fully_connected_layer_sizes_[i]);
  (load_file_.empty())? InitializeBiasesWeightsKernels(s_shaped_vectors) : LoadBiasesWeightsKernels();
  std::cout << "- Done." << '\n' << "Selected hyper-parameters: " << '\n' << '\t' << "Activation function in convolutional layers: " << ((leaky_ReLU_in_convolutional_layers_)? "leaky ReLU" : "ReLU") << '\n' << '\t' << "Activation function in fully-connected layers: ";
  switch (activation_function_in_fully_connected_layers_) {
   case 0:
    std::cout << "ReLu";
    break;
   case 1:
    std::cout << "leaky ReLU";
    break;
   case 2:
    std::cout << "Sigmoid";
    break;
   default:
    std::cout << "tanh";
  }
  std::cout << '\n' << '\t' << "Number of convolutional and pooling layers: " << num_of_convolutional_and_pooling_layers_ << '\n' << '\t' << "Number of filters: " << num_of_feature_maps_ << '\n' << '\t' << "Kernel sizes (in convolutional layers): " << kernel_sizes_[0] << "/" << kernel_sizes_[1] << '\n' << '\t' << "Pool-size: 2/2" << '\n' << '\t' << "Pooling: ";
  switch (pooling_) {
   case 0:
    std::cout << "max-pooling";
    break;
   case 1:
    std::cout << "L2-pooling";
    break;
   default:
    std::cout << "average-pooling";
  }
  std::cout << '\n' << '\t' << "Stride: 1" << '\n' << '\t' << "Padding: zero padding" << '\n' << '\t' << "No biases will be used in the convolutional layers!" << '\n' << '\t' << "Sizes of fully-connected layers: ";
  for (unsigned i = 0; i < fully_connected_layer_sizes_.size(); ++i)
    std::cout << fully_connected_layer_sizes_[i] << ((i == (fully_connected_layer_sizes_.size()-1))? '\n' : '/');
  std::cout << '\t' << "Epochs: " << epochs_ << '\n' << '\t' << "Mini-batch size: " << mini_batch_size_ << '\n' << '\t' << "Lambda: " << lambda_ << '\n' << '\t' << "Learning rate (eta): " << learning_rate_ << '\n' << "Initial test started..." << std::endl;
  if (test_size_ != 0) Evaluate(-1); // tests the initialized network before
                                     // the training has started
  std::cout << "Training started..." << std::endl;
  StochasticGradientDescent();
}

ConvolutionalNetwork::ConvolutionalNetwork(const std::vector<std::vector<std::vector<double>>>& input, const std::vector<std::vector<std::string>>& words, const std::string& save_directory, const std::string& load_file, const unsigned pooling)
    : NeuralNetwork(input.size(), words, save_directory, load_file),
      train_input_(input),
      test_input_({{}}),
      fully_connected_layer_sizes_(GetFullyConnectedLayerSizes(load_file)),
      num_of_feature_maps_(GetNumOfFeatureMaps(load_file)),
      num_of_convolutional_and_pooling_layers_(GetNumOfConvolutionalLayers(load_file)),
      num_of_fully_connected_layers_(fully_connected_layer_sizes_.size()),
      kernel_sizes_(GetKernelSizes(load_file)),
      even_kernel_sizes_(((((kernel_sizes_[0]-1)/2)%2)==0), ((((kernel_sizes_[1]-1)/2)%2)==0)),
      pooling_(pooling),
      leaky_ReLU_in_convolutional_layers_(GetActivationFunctionOfConvolutionalLayers(load_file)) {
  std::cout << "- Done." << '\n' << "Loading network..." << std::endl;
  convolutional_layers_.resize(1+2*num_of_convolutional_and_pooling_layers_);
  convolutional_layers_[0].resize(1);
  convolutional_layers_[0][0] = std::vector<std::vector<double>>(train_input_[0].size(), std::vector<double>(train_input_[0][0].size()));
  for (unsigned i = 0; i < num_of_convolutional_and_pooling_layers_; ++i) {
    convolutional_layers_[2*i+1].resize(num_of_feature_maps_);
    convolutional_layers_[2*i+2].resize(num_of_feature_maps_);
    for (unsigned j = 0; j < num_of_feature_maps_; ++j) {
      convolutional_layers_[2*i+1][j] = convolutional_layers_[2*i][0]; // convolutional layer
      convolutional_layers_[2*i+2][j] = std::vector<std::vector<double>>((((convolutional_layers_[2*i+1][0].size()%2) == 0)? convolutional_layers_[2*i+1][0].size()/2+1 : (convolutional_layers_[2*i+1][0].size()+1)/2+1), std::vector<double>((((convolutional_layers_[2*i+1][0][0].size()%2) == 0)? convolutional_layers_[2*i+1][0][0].size()/2+1 : (convolutional_layers_[2*i+1][0][0].size()+1)/2+1))); // pooling layer
    }
  }
  convolutional_netinputs_.resize(num_of_convolutional_and_pooling_layers_);
  for (unsigned i = 0; i < num_of_convolutional_and_pooling_layers_; ++i)
    convolutional_netinputs_[i] = convolutional_layers_[2*i+1];
  fully_connected_layers_.resize(num_of_fully_connected_layers_);
  for (unsigned i = 0; i < num_of_fully_connected_layers_; ++i)
    fully_connected_layers_[i].resize(fully_connected_layer_sizes_[i]);
  LoadBiasesWeightsKernels();
  std::cout << "- Done." << '\n' << "Starting to classify the inputs..." << std::endl;
  OnDuty();
  std::cout << "- Done. Classifications have been saved." << '\n' << "Program terminated." << std::endl;
}

void ConvolutionalNetwork::InitializeBiasesWeightsKernels(const bool s_shaped_vectors) {
// Initializes the values of the biases, weights and kernals randomly.
  s_shaped_vectors_ = s_shaped_vectors;
  std::random_device ran_dev {};
  std::mt19937 generator {ran_dev()};
  std::normal_distribution <> distribution_biases {0, 1};
  std::normal_distribution <> distribution_kernels {0, (1/(1/std::sqrt(kernel_sizes_[0]*kernel_sizes_[1])))};
  kernels_.resize(num_of_convolutional_and_pooling_layers_);
  nabla_kernels_.resize(num_of_convolutional_and_pooling_layers_);
  for (unsigned i = 0; i < num_of_convolutional_and_pooling_layers_; ++i) {
    kernels_[i].resize(num_of_feature_maps_);
    nabla_kernels_[i].resize(num_of_feature_maps_);
    for (unsigned j = 0; j < num_of_feature_maps_; ++j) {
      kernels_[i][j].resize(kernel_sizes_[0]);
      nabla_kernels_[i][j].resize(kernel_sizes_[0]);
      for (unsigned k = 0; k < kernel_sizes_[0]; ++k) {
        kernels_[i][j][k].resize(kernel_sizes_[1]);
        nabla_kernels_[i][j][k].resize(kernel_sizes_[1]);
        for (unsigned l = 0; l < kernel_sizes_[1]; ++l)
          kernels_[i][j][k][l] = distribution_kernels(generator);
      }
    }
  }
  biases_.resize(num_of_fully_connected_layers_);
  for (unsigned i = 0; i < num_of_fully_connected_layers_; ++i)
    biases_[i].resize(fully_connected_layer_sizes_[i]);
  flattening_weights_.resize(fully_connected_layer_sizes_[0]);
  flattening_nabla_w_ = flattening_weights_;
  std::normal_distribution <> distribution_flattening_weights {0, (1/std::sqrt(convolutional_layers_[2*num_of_convolutional_and_pooling_layers_].size()*convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][0].size()))};
  for (unsigned i = 0; i < fully_connected_layer_sizes_[0]; ++i) {
    biases_[0][i] = distribution_biases(generator);
    flattening_weights_[i].resize(num_of_feature_maps_);
    flattening_nabla_w_[i] = flattening_weights_[i];
    for (unsigned j = 0; j < num_of_feature_maps_; ++j) {
      std::vector<std::vector<double>>& current_flattening_weights = flattening_weights_[i][j];
      std::vector<std::vector<double>>& current_flattening_nabla_w = flattening_nabla_w_[i][j];
      current_flattening_weights.resize(convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][0].size());
      current_flattening_nabla_w = current_flattening_weights;
      for (unsigned k = 0; k < convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][0].size(); ++k) {
        current_flattening_weights[k].resize(convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][0][0].size());
        current_flattening_nabla_w[k] = current_flattening_weights[k];
        for (auto& current_flattening_weight : current_flattening_weights[k])
          current_flattening_weight = distribution_flattening_weights(generator);
      }
    }
  }
  weights_.resize(num_of_fully_connected_layers_-1);
  nabla_w_ = weights_;
  for (unsigned i = 1; i < num_of_fully_connected_layers_; ++i) {
    biases_[i].resize(fully_connected_layer_sizes_[i]);
    weights_[i-1].resize(fully_connected_layer_sizes_[i]);
    nabla_w_[i-1] = weights_[i-1];
    std::normal_distribution <> distribution_weights {0, (1/std::sqrt(fully_connected_layer_sizes_[i-1]))};
    for (unsigned j = 0; j < fully_connected_layer_sizes_[i]; ++j) {
      biases_[i][j] = distribution_biases(generator);
      std::vector<double>& temp_weights = weights_[i-1][j];
      temp_weights.resize(fully_connected_layer_sizes_[i-1]);
      nabla_w_[i-1][j] = temp_weights;
      for (unsigned k = 0; k < fully_connected_layer_sizes_[i-1]; ++k)
        temp_weights[k] = distribution_weights(generator);
    }
  }
}

void ConvolutionalNetwork::LoadBiasesWeightsKernels() {
  std::ifstream load_file_stream(load_file_);
  std::string line, value;
  std::getline(load_file_stream, line); // skips the activation function of the
                                        // fully connected layers, which is
                                        // already initialized
  std::getline(load_file_stream, line);
  s_shaped_vectors_ = std::stoi(line);
  std::getline(load_file_stream, line);
  if (leaky_ReLU_in_convolutional_layers_ != std::stoi(line))
    std::cout << "WARNING: The activation function you have chosen for the convolutional layers is not equal to the activation function found in the file of the network you want to load! Errors may occur!" << std::endl;
  std::getline(load_file_stream, line);
  if (num_of_convolutional_and_pooling_layers_ != ((unsigned) std::stoi(line)))
    std::cout << "WARNING: The number of convolutional and pooling layers you have chosen is not equal to the number of convolutional and pooling layers found in the file of the network you want to load! Errors may occur!" << std::endl;
  std::getline(load_file_stream, line);
  if (num_of_feature_maps_ != ((unsigned) std::stoi(line)))
    std::cout << "WARNING: The number of filters you have chosen is not equal to the number of filters found in the file of the network you want to load! Errors may occur!" << std::endl;
  std::getline(load_file_stream, line);
  std::stringstream stream_kernel_sizes(line);
  for (unsigned i = 0; i < 2; ++i) {
    getline(stream_kernel_sizes, value, ' ');
    if (kernel_sizes_[0] != ((unsigned) std::stoi(value))) {
      std::cout << "WARNING: The kernel sizes you have chosen is not equal to the kernel sizes found in the file of the network you want to load! Errors may occur!" << std::endl;
      break;
    }
  }
  std::getline(load_file_stream, line);
  std::stringstream stream_fully_connected_layer_sizes(line);
  std::vector<unsigned> check_vector_for_fully_connected_layer_sizes;
  while (getline(stream_fully_connected_layer_sizes, value, ' '))
    check_vector_for_fully_connected_layer_sizes.push_back(std::stoi(value));
  if (fully_connected_layer_sizes_ != check_vector_for_fully_connected_layer_sizes)
    std::cout << "WARNING: The fully-connected layer sizes you have chosen are not equal to the fully-connected layer sizes found in the file of the network you want to load! Errors may occur!" << std::endl;
  kernels_.resize(num_of_convolutional_and_pooling_layers_);
  nabla_kernels_.resize(num_of_convolutional_and_pooling_layers_);
  for (unsigned i = 0; i < num_of_convolutional_and_pooling_layers_; ++i) {
    kernels_[i].resize(num_of_feature_maps_);
    nabla_kernels_[i].resize(num_of_feature_maps_);
    for (unsigned j = 0; j < num_of_feature_maps_; ++j) {
      kernels_[i][j].resize(kernel_sizes_[0]);
      nabla_kernels_[i][j].resize(kernel_sizes_[0]);
      std::getline(load_file_stream, line);
      std::stringstream stream_kernels(line);
      for (unsigned k = 0; k < kernel_sizes_[0]; ++k) {
        kernels_[i][j][k].resize(kernel_sizes_[1]);
        nabla_kernels_[i][j][k].resize(kernel_sizes_[1]);
        for (auto& weight : kernels_[i][j][k]) {
          getline(stream_kernels, value, ' ');
          weight = atof(value.c_str());
        }
      }
    }
  }
  flattening_weights_.resize(fully_connected_layer_sizes_[0]);
  flattening_nabla_w_ = flattening_weights_;
  for (unsigned i = 0; i < flattening_weights_.size(); ++i) {
    flattening_weights_[i].resize(num_of_feature_maps_);
    flattening_nabla_w_[i] = flattening_weights_[i];
    std::getline(load_file_stream, line);
    std::stringstream stream_flattening_weights(line);
    for (unsigned j = 0; j < flattening_weights_[i].size(); ++j) {
      std::vector<std::vector<double>>& current_flattening_weights = flattening_weights_[i][j];
      std::vector<std::vector<double>>& current_flattening_nabla_w = flattening_nabla_w_[i][j];
      current_flattening_weights.resize(convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][0].size());
      current_flattening_nabla_w = current_flattening_weights;
      for (unsigned k = 0; k < flattening_weights_[i][j].size(); ++k) {
        current_flattening_weights[k].resize(convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][0][0].size());
        current_flattening_nabla_w[k] = current_flattening_weights[k];
        for (auto& flattening_weight : flattening_weights_[i][j][k]) {
          getline(stream_flattening_weights, value, ' ');
          flattening_weight = atof(value.c_str());
        }
      }
    }
  }
  weights_.resize(num_of_fully_connected_layers_-1);
  nabla_w_ = weights_;
  for (unsigned i = 1; i < num_of_fully_connected_layers_; ++i) {
    weights_[i-1].resize(fully_connected_layer_sizes_[i]);
    nabla_w_[i-1] = weights_[i-1];
    std::getline(load_file_stream, line);
    std::stringstream stream_weights(line);
    for (unsigned j = 0; j < fully_connected_layer_sizes_[i]; ++j) {
      std::vector<double>& temp_weights = weights_[i-1][j];
      temp_weights.resize(fully_connected_layer_sizes_[i-1]);
      nabla_w_[i-1][j] = temp_weights;
      for (unsigned k = 0; k < fully_connected_layer_sizes_[i-1]; ++k) {
        getline(stream_weights, value, ' ');
        temp_weights[k] = atof(value.c_str());
      }
    }
  }
  std::getline(load_file_stream, line);
  std::stringstream stream_biases(line);
  biases_.resize(num_of_fully_connected_layers_);
  for (unsigned i = 0; i < num_of_fully_connected_layers_; ++i) {
    biases_[i].resize(fully_connected_layer_sizes_[i]);
    for (auto& bias : biases_[i]) {
      getline(stream_biases, value, ' ');
      bias = atof(value.c_str());
    }
  }
}

void ConvolutionalNetwork::StochasticGradientDescent() {
  const unsigned number_of_mini_batches = train_size_/mini_batch_size_;
  std::vector<unsigned> mini_batch_indices(train_size_);
  std::iota(mini_batch_indices.begin(), mini_batch_indices.end(), 0);
  for (unsigned epoch = 0; epoch < epochs_; ++epoch) {
    std::random_shuffle(mini_batch_indices.begin(), mini_batch_indices.end(), GetRandomInt); // shuffles the mini batches
    for (unsigned mini_batch = 0; mini_batch < number_of_mini_batches; ++mini_batch)
      UpdateMiniBatches(mini_batch, mini_batch_indices, (mini_batch == (number_of_mini_batches-1)));
    if (test_size_ != 0)
      Evaluate(epoch);
    else
      std::cout << (epoch+1) << ". epoch completed." << std::endl;
  }
  if (test_size_ == 0 && !save_file_.empty())
  // If no test data is provided biases and weights will be saved after the
  // last epoch (otherwise only those biases and weights that led to the best
  // test result will be saved).
    SaveWeights();
  std::cout << "Training finished." << std::endl;
}

void ConvolutionalNetwork::UpdateMiniBatches(const unsigned mini_batch, const std::vector<unsigned>& mini_batch_indices, const bool last_mini_batch) {
  const unsigned& current_mini_batch_size = (last_mini_batch && mini_batch_size_remainder_ != 0)? mini_batch_size_remainder_ : mini_batch_size_;
  std::vector<std::vector<std::vector<double>>> all_fully_connected_nabla_b(current_mini_batch_size, std::vector<std::vector<double>>(num_of_fully_connected_layers_));
  std::vector<std::vector<std::vector<std::vector<double>>>> all_nabla_w(current_mini_batch_size, std::vector<std::vector<std::vector<double>>>(num_of_fully_connected_layers_-1));
  std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> all_flattening_nabla_w(current_mini_batch_size, std::vector<std::vector<std::vector<std::vector<double>>>>(num_of_feature_maps_, std::vector<std::vector<std::vector<double>>>(fully_connected_layer_sizes_[0])));
  std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> all_nabla_kernels(current_mini_batch_size, std::vector<std::vector<std::vector<std::vector<double>>>>(num_of_convolutional_and_pooling_layers_, std::vector<std::vector<std::vector<double>>>(num_of_feature_maps_)));
  unsigned i = 0;
  for (unsigned j = (mini_batch*mini_batch_size_); j < (mini_batch*mini_batch_size_+current_mini_batch_size); ++j) {
    Backpropagate(mini_batch_indices[j]);
    all_nabla_kernels[i] = nabla_kernels_;
    all_fully_connected_nabla_b[i] = nabla_b_;
    all_nabla_w[i] = nabla_w_;
    all_flattening_nabla_w[i] = flattening_nabla_w_;
    i++;
  }
  // Calculates the new biases and weights of the fully connected layers.
  for (unsigned i = 1; i < current_mini_batch_size; ++i) {
    for (unsigned j = 0; j < num_of_fully_connected_layers_; ++j) {
      std::transform(all_fully_connected_nabla_b[0][j].begin(), all_fully_connected_nabla_b[0][j].end(), all_fully_connected_nabla_b[i][j].begin(), all_fully_connected_nabla_b[0][j].begin(), std::plus<double>());
      if (j != (num_of_fully_connected_layers_-1)) {
        const std::vector<std::vector<double>>& current_nabla_w = all_nabla_w[i][j];
        std::vector<std::vector<double>>& current_nabla_w_sum = all_nabla_w[0][j];
        for (unsigned k = 0; k < fully_connected_layer_sizes_[j+1]; ++k)
          std::transform(current_nabla_w_sum[k].begin(), current_nabla_w_sum[k].end(), current_nabla_w[k].begin(), current_nabla_w_sum[k].begin(), std::plus<double>());
      }
    }
    for (unsigned j = 0; j < fully_connected_layer_sizes_[0]; ++j) {
      for (unsigned k = 0; k < num_of_feature_maps_; ++k) {
        const std::vector<std::vector<double>>& current_nabla_w = all_flattening_nabla_w[i][j][k];
        std::vector<std::vector<double>>& current_nabla_w_sum = all_flattening_nabla_w[0][j][k];
        for (unsigned l = 0; l < flattening_nabla_w_[0][0].size(); ++l)
          std::transform(current_nabla_w_sum[l].begin(), current_nabla_w_sum[l].end(), current_nabla_w[l].begin(), current_nabla_w_sum[l].begin(), std::plus<double>());
      }
    }
    for (unsigned j = 0; j < num_of_convolutional_and_pooling_layers_; ++j) {
      for (unsigned k = 0; k < num_of_feature_maps_; ++k) {
        const std::vector<std::vector<double>>& current_nabla_kernels = all_nabla_kernels[i][j][k];
        std::vector<std::vector<double>>& current_nabla_kernels_sum = all_nabla_kernels[0][j][k];
        for (unsigned l = 0; l < kernel_sizes_[0]; ++l)
          std::transform(current_nabla_kernels_sum[l].begin(), current_nabla_kernels_sum[l].end(), current_nabla_kernels[l].begin(), current_nabla_kernels_sum[l].begin(), std::plus<double>());
      }
    }
  }
  const double scalar0 = learning_rate_/current_mini_batch_size;
  const double scalar1 = 1-learning_rate_*(lambda_/train_size_); // used for regularization
  for (unsigned i = 0; i < num_of_fully_connected_layers_; ++i) {
    std::transform(all_fully_connected_nabla_b[0][i].begin(), all_fully_connected_nabla_b[0][i].end(), all_fully_connected_nabla_b[0][i].begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, scalar0));
    std::transform(biases_[i].begin(), biases_[i].end(), all_fully_connected_nabla_b[0][i].begin(), biases_[i].begin(), std::minus<double>());
    if (i != (num_of_fully_connected_layers_-1)) {
      std::vector<std::vector<double>>& nabla_w_sum = all_nabla_w[0][i];
      std::vector<std::vector<double>>& current_weights = weights_[i];
      for (unsigned j = 0; j < fully_connected_layer_sizes_[i+1]; ++j) {
        std::transform(nabla_w_sum[j].begin(), nabla_w_sum[j].end(), nabla_w_sum[j].begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, scalar0));
        std::transform(current_weights[j].begin(), current_weights[j].end(), current_weights[j].begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, scalar1));
        std::transform(current_weights[j].begin(), current_weights[j].end(), nabla_w_sum[j].begin(), current_weights[j].begin(), std::minus<double>());
      }
    }
  }
  // Backpropagates from the (one-dimensional) first fully connected layer to
  // the last of the (two-dimensional) convolutional or rather pooling layers.
  for (unsigned i = 0; i < fully_connected_layer_sizes_[0]; ++i) {
    for (unsigned j = 0; j < num_of_feature_maps_; ++j) {
      std::vector<std::vector<double>>& nabla_w_sum = all_flattening_nabla_w[0][i][j];
      std::vector<std::vector<double>>& current_weights = flattening_weights_[i][j];
      for (unsigned k = 0; k < flattening_nabla_w_[0][0].size(); ++k) {
        std::transform(nabla_w_sum[k].begin(), nabla_w_sum[k].end(), nabla_w_sum[k].begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, scalar0));
        std::transform(current_weights[k].begin(), current_weights[k].end(), current_weights[k].begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, scalar1));
        std::transform(current_weights[k].begin(), current_weights[k].end(), nabla_w_sum[k].begin(), current_weights[k].begin(), std::minus<double>());
      }
    }
  }
  // Calculates the new biases and weights of the convolutional and pooling
  // layers.
  for (unsigned i = 0; i < num_of_convolutional_and_pooling_layers_; ++i) {
    for (unsigned j = 0; j < num_of_feature_maps_; ++j) {
      std::vector<std::vector<double>>& nabla_kernels_sum = all_nabla_kernels[0][i][j];
      std::vector<std::vector<double>>& current_kernel = kernels_[i][j];
      for (unsigned k = 0; k < kernel_sizes_[0]; ++k) {
        std::transform(nabla_kernels_sum[k].begin(), nabla_kernels_sum[k].end(), nabla_kernels_sum[k].begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, scalar0));
        std::transform(current_kernel[k].begin(), current_kernel[k].end(), current_kernel[k].begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, scalar1));
        std::transform(current_kernel[k].begin(), current_kernel[k].end(), nabla_kernels_sum[k].begin(), current_kernel[k].begin(), std::minus<double>());
      }
    }
  }
}

void ConvolutionalNetwork::Backpropagate(const unsigned train_data_index) {
  Feedforward(train_data_index, false);
  // Calculates nablas for the fully connected sub-network.
  std::vector<std::vector<double>> fully_connected_deltas(num_of_fully_connected_layers_);
  for (unsigned i = 0; i < num_of_fully_connected_layers_; ++i)
    fully_connected_deltas[i].resize(fully_connected_layer_sizes_[i]);
  std::transform(fully_connected_layers_[num_of_fully_connected_layers_-1].begin(), fully_connected_layers_[num_of_fully_connected_layers_-1].end(), train_output_[train_data_index].begin(), fully_connected_deltas[num_of_fully_connected_layers_-1].begin(), std::minus<double>());
  for (unsigned i = (num_of_fully_connected_layers_-1); i > 0; --i) {
    std::fill(fully_connected_deltas[i-1].begin(), fully_connected_deltas[i-1].end(), 0);
    for (unsigned j = 0; j < fully_connected_layer_sizes_[i-1]; ++j) {
      for (unsigned k = 0; k < fully_connected_layer_sizes_[i]; ++k)
        fully_connected_deltas[i-1][j] += fully_connected_deltas[i][k]*weights_[i-1][k][j];
      fully_connected_deltas[i-1][j] *= DerivatedActivationFunctionInFullyConnectedLayers(fully_connected_netinputs_[i-1][j]);
    }
  }
  nabla_b_ = fully_connected_deltas;
  for (unsigned i = 0; i < (num_of_fully_connected_layers_-1); ++i) {
    std::vector<std::vector<double>>& nw_i = nabla_w_[i];
    for (unsigned j = 0; j < fully_connected_layer_sizes_[i+1]; ++j) {
      const double& current_delta = fully_connected_deltas[i+1][j];
      for (unsigned k = 0; k < fully_connected_layer_sizes_[i]; ++k)
        nw_i[j][k] = current_delta*fully_connected_layers_[i][k];
    }
  }
  // Calculates nablas for the weights connecting the convolutional and the fully connected sub-networks.
  for (unsigned i = 0; i < fully_connected_layer_sizes_[0]; ++i) {
    const double& current_delta = fully_connected_deltas[0][i];
    for (unsigned j = 0; j < num_of_feature_maps_; ++j) {
      std::vector<std::vector<double>>& fnw_ij = flattening_nabla_w_[i][j];
      for (unsigned k = 0; k < (convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][0].size()-(kernel_sizes_[0]-1)); ++k) {
        for (unsigned l = 0; l < (convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][0][0].size()-(kernel_sizes_[1]-1)); ++l)
          fnw_ij[k][l] = current_delta*convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][j][k+1][l+1];
      }
    }
  }
  // Calculates deltas for the convolutional layers.
  std::vector<std::vector<std::vector<std::vector<double>>>> convolutional_deltas(num_of_convolutional_and_pooling_layers_, std::vector<std::vector<std::vector<double>>>(num_of_feature_maps_));
  for (unsigned i = 0; i < num_of_feature_maps_; ++i) {
    convolutional_deltas[num_of_convolutional_and_pooling_layers_-1][i].resize(convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][0].size()-(kernel_sizes_[0]-1));
    for (unsigned j = 0; j < (convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][0].size()-(kernel_sizes_[0]-1)); ++j) {
      convolutional_deltas[num_of_convolutional_and_pooling_layers_-1][i][j].resize(convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][0][0].size()-(kernel_sizes_[1]-1));
      std::fill(convolutional_deltas[num_of_convolutional_and_pooling_layers_-1][i][j].begin(), convolutional_deltas[num_of_convolutional_and_pooling_layers_-1][i][j].end(), 0);
      for (unsigned k = 0; k < (convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][0][0].size()-(kernel_sizes_[1]-1)); ++k) {
        for (unsigned l = 0; l < fully_connected_layer_sizes_[0]; ++l) {
          for (unsigned m = 0; m < kernel_sizes_[0]; ++m) {
            for (unsigned n = 0; n < kernel_sizes_[1]; ++n)
              convolutional_deltas[num_of_convolutional_and_pooling_layers_-1][i][j][k] += fully_connected_deltas[0][l]*kernels_[num_of_convolutional_and_pooling_layers_-1][i][m][n];
          }
        }
        convolutional_deltas[num_of_convolutional_and_pooling_layers_-1][i][j][k] *= DerivatedActivationFunctionInConvolutionalLayers(convolutional_netinputs_[num_of_convolutional_and_pooling_layers_-1][i][j][k]);
      }
    }
  }
  for (unsigned i = (num_of_convolutional_and_pooling_layers_-1); i > 0; --i) {
    for (unsigned j = 0; j < num_of_feature_maps_; ++j) {
      convolutional_deltas[i-1][j].resize(convolutional_layers_[2*i][0].size()-(kernel_sizes_[0]-1));
      for (unsigned k = 0; k < convolutional_layers_[2*i][0].size()-(kernel_sizes_[0]-1); ++k) {
        convolutional_deltas[i-1][j][k].resize(convolutional_layers_[2*i][0][0].size()-(kernel_sizes_[1]-1));
        std::fill(convolutional_deltas[i-1][j][k].begin(), convolutional_deltas[i-1][j][k].end(), 0);
        for (unsigned l = 0; l < (convolutional_layers_[2*i][0][0].size()-(kernel_sizes_[1]-1)); ++l) {
          for (unsigned m = 0; m < (convolutional_layers_[2*i+2][0].size()-(kernel_sizes_[0]-1)); ++m) {
            for (unsigned n = 0; n < (convolutional_layers_[2*i+2][0][0].size()-(kernel_sizes_[1]-1)); ++n) {
              for (unsigned o = 0; o < kernel_sizes_[0]; ++o) {
                for (unsigned p = 0; p < kernel_sizes_[1]; ++p)
                  convolutional_deltas[i-1][j][k][l] += convolutional_deltas[i][j][m][n]*kernels_[i-1][j][o][p];
              }
            }
          }
          convolutional_deltas[i-1][j][k][l] *= DerivatedActivationFunctionInConvolutionalLayers(convolutional_netinputs_[i-1][j][k][l]);
        }
      }
    }
  }
  // Calculates nablas for the convolutional layers.
  for (unsigned i = 0; i < num_of_convolutional_and_pooling_layers_; ++i) {
    for (unsigned j = 0; j < num_of_feature_maps_; ++j) {
      const std::vector<std::vector<double>>& current_deltas = convolutional_deltas[i][j];
      const std::vector<std::vector<double>>& current_layer = convolutional_layers_[2*i+1][j];
      for (unsigned k = 0; k < kernel_sizes_[0]; ++k) {
        for (unsigned l = 0; l < kernel_sizes_[1]; ++l) {
          std::fill(nabla_kernels_[i][j][k].begin(), nabla_kernels_[i][j][k].end(), 0);
          double& current_nabla_kernel = nabla_kernels_[i][j][k][l];
          for (unsigned m = 0; m < current_deltas.size(); ++m) {
            for (unsigned n = 0; n < current_deltas[0].size(); ++n)
              current_nabla_kernel += current_deltas[m][n]*current_layer[m+k][n+l];
          }
        }
      }
    }
  }
}

void ConvolutionalNetwork::Feedforward(const unsigned data_index, const bool test) {
  // Feedforward through the convolutional and pooling layers.
  convolutional_layers_[0][0] = (test)? test_input_[data_index] : train_input_[data_index];
  for (unsigned i = 0; i < num_of_convolutional_and_pooling_layers_; ++i) {
    for (unsigned j = 0; j < num_of_feature_maps_; ++j) {
      const std::vector<std::vector<double>>& current_kernel = kernels_[i][j];
      const std::vector<std::vector<double>>& current_layer = convolutional_layers_[2*i][((i == 0)? 0 : j)];
      std::vector<std::vector<double>>& netinputs_to_calculate = convolutional_netinputs_[i][j];
      std::vector<std::vector<double>>& layer_to_caluclate = convolutional_layers_[2*i+1][j];
      for (unsigned k = (kernel_sizes_[0]-1)/2; k < (convolutional_layers_[2*i][0].size()-(kernel_sizes_[0]-1)/2); ++k) {
        for (unsigned l = (kernel_sizes_[1]-1)/2; l < (convolutional_layers_[2*i][0][0].size()-(kernel_sizes_[1]-1)/2); ++l) {
          netinputs_to_calculate[k-1][l-1] = 0;
          for (unsigned m = 0; m < kernel_sizes_[0]; ++m) {
            for (unsigned n = 0; n < kernel_sizes_[1]; ++n)
              netinputs_to_calculate[k-1][l-1] += current_layer[k+(m-(kernel_sizes_[0]-1)/2)][l+(n-(kernel_sizes_[1]-1)/2)]*current_kernel[m][n];
          }
          layer_to_caluclate[k][l] = ActivationFunctionInConvolutionalLayers(netinputs_to_calculate[k-1][l-1]);
        }
      }
    }
    Pooling(i);
  }
  // Connecting last pooling layer with first fully connected layer.
  fully_connected_netinputs_ = biases_;
  for (unsigned i = 0; i < fully_connected_layer_sizes_[0]; ++i) {
    for (unsigned j = 0; j < num_of_feature_maps_; ++j) {
      const std::vector<std::vector<double>>& fw_ij = flattening_weights_[i][j];
      const std::vector<std::vector<double>>& current_feature_map = convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][j];
      for (unsigned k = 0; k < (convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][0].size()-2); ++k) {
        for (unsigned l = 0; l < (convolutional_layers_[2*num_of_convolutional_and_pooling_layers_][0][0].size()-2); ++l)
          fully_connected_netinputs_[0][i] += fw_ij[k][l]*current_feature_map[k+1][l+1]; // flattening, connecting convolutional layers with the fully connected ones
      }
    }
    fully_connected_layers_[0][i] = ActivationFunctionInFullyConnectedLayers(fully_connected_netinputs_[0][i]);
  }
  // Feedforward through the fully connected layers.
  for (unsigned i = 1; i < num_of_fully_connected_layers_; ++i) {
    for (unsigned j = 0; j < fully_connected_layer_sizes_[i]; ++j) {
      const std::vector<double>& current_weights = weights_[i-1][j];
      double& current_netinput = fully_connected_netinputs_[i][j];
      for (unsigned k = 0; k < fully_connected_layer_sizes_[i-1]; ++k)
        current_netinput += current_weights[k]*fully_connected_layers_[i-1][k];
    }
    if (i != (num_of_fully_connected_layers_-1)) {
      for (unsigned j = 0; j < fully_connected_layer_sizes_[i]; ++j)
        fully_connected_layers_[i][j] = ActivationFunctionInFullyConnectedLayers(fully_connected_netinputs_[i][j]);
    } else {
      std::vector<double>& output_netinputs = fully_connected_netinputs_[num_of_fully_connected_layers_-1];
      if (activation_function_in_fully_connected_layers_ != 2) { // softmax function will be used in the output layer
        // Softmax:
        double sum = 0;
        for (auto& netinput : output_netinputs)
          sum += std::exp(netinput);
        for(unsigned i = 0; i < fully_connected_layer_sizes_[num_of_fully_connected_layers_-1]; ++i)
          fully_connected_layers_[num_of_fully_connected_layers_-1][i] = std::exp(output_netinputs[i])/sum;
      } else { // sigmoid will be used in the output layer
        // Sigmoid:
        for (unsigned i = 0; i < fully_connected_layer_sizes_[num_of_fully_connected_layers_-1]; ++i)
          fully_connected_layers_[num_of_fully_connected_layers_-1][i] = ActivationFunctionInFullyConnectedLayers(output_netinputs[i]);
      }
    }
  }
}

void ConvolutionalNetwork::Pooling(const unsigned i) {
  unsigned m, n;
  std::vector<double> pool(4);
  for (unsigned j = 0; j < num_of_feature_maps_; ++j) {
    std::vector<std::vector<double>>& current_layer = convolutional_layers_[2*i+1][j];
    std::vector<std::vector<double>>& layer_to_calculate = convolutional_layers_[2*i+2][j];
    m = (kernel_sizes_[0]-1)/2;
    for (unsigned k = (kernel_sizes_[0]-1)/2; k < (current_layer.size()-(kernel_sizes_[0]-1)/2); (k == (((current_layer.size()-(kernel_sizes_[0]-1)/2)-1) && ((!even_kernel_sizes_[0] && (current_layer.size()%2) != 0) || (even_kernel_sizes_[0] && (current_layer.size()%2) == 0)))? ++k : k=k+2)) {
      n = (kernel_sizes_[1]-1)/2;
      for (unsigned l = (kernel_sizes_[1]-1)/2; l < (current_layer[k].size()-(kernel_sizes_[1]-1)/2); (l == (((current_layer[k].size()-(kernel_sizes_[1]-1)/2)-1) && ((!even_kernel_sizes_[1] && (current_layer[k].size()%2) != 0) || (even_kernel_sizes_[1] && (current_layer[k].size()%2) == 0)))? ++l : l=l+2)) {
        if (pooling_ == 0) { // max pooling
          if ((!even_kernel_sizes_[0] && ((k%2) != 0) && !even_kernel_sizes_[1] && ((l%2) != 0)) || (even_kernel_sizes_[0] && ((k%2) == 0) && even_kernel_sizes_[1] && ((l%2) == 0)))
            pool = {current_layer[k][l], current_layer[k+1][l], current_layer[k][l+1], current_layer[k+1][l+1]};
          else if ((!even_kernel_sizes_[0] && ((k%2) != 0)) || (even_kernel_sizes_[0] && ((k%2) == 0)))
            pool = {current_layer[k][l], current_layer[k+1][l], 0, 0};
          else if ((!even_kernel_sizes_[1] && ((l%2) != 0)) || (even_kernel_sizes_[1] && ((l%2) == 0)))
            pool = {current_layer[k][l], current_layer[k][l+1], 0, 0};
          else
            pool = {current_layer[k][l], 0, 0, 0};
          layer_to_calculate[m][n] = *std::max_element(pool.begin(), pool.end());
        } else {
          layer_to_calculate[m][n] = 0;
          if ((k%2) != 0 && (l%2) != 0)
            layer_to_calculate[m][n] = current_layer[k][l]+current_layer[k+1][l]+current_layer[k][l+1]+current_layer[k+1][l+1];
          else if ((k%2) != 0)
            layer_to_calculate[m][n] = current_layer[k][l]+current_layer[k+1][l];
          else if ((l%2) != 0)
            layer_to_calculate[m][n] = current_layer[k][l]+current_layer[k][l+1];
          else
            layer_to_calculate[m][n] = current_layer[k][l];
          if (pooling_ == 1) // L2 pooling
            layer_to_calculate[m][n] = std::sqrt(layer_to_calculate[m][n]);
          else // average pooling
            layer_to_calculate[m][n] = layer_to_calculate[m][n]/4;
        }
        n++;
      }
      m++;
    }
  }
}

void ConvolutionalNetwork::Evaluate(const int epoch) {
  unsigned correct = 0, wrong = 0;
  std::vector<std::string> wrongly_classified_words;
  wrongly_classified_words.reserve(100);
  for (unsigned i = 0; i < test_size_; ++i) {
    Feedforward(i, true);
    std::vector<double>& output_layer = fully_connected_layers_[num_of_fully_connected_layers_-1];
    if (test_output_[i] == std::distance(output_layer.begin(), std::max_element(output_layer.begin(), output_layer.end())))
      correct++;
    else if (epoch == ((int) epochs_-1) && wrong < 100) {
      wrongly_classified_words.push_back(words_[1][i]);
      wrong++;
    }
  }
  std::cout << (epoch+1) << ". epoch, correct results: " << '\n' << '\t' << "Correct results: " << correct << '/' << test_size_ << " (accuracy: " << ((double) correct/(test_size_/100)) << "%)" << std::endl;
  if (!save_file_.empty() && correct > best_test_result) {
    best_test_result = correct;
    if (epoch != -1)
      SaveWeights();
  }
  if (epoch == ((int) epochs_-1) && wrong != 0) {
    std::cout << "Words that were classified wrongly: " << '\n'; // shows not more than 100 wrongly classified words
    for (auto& wrong_word : wrongly_classified_words)
      std::cout << ' ' << wrong_word;
    std::cout << '\n';
  }
}

void ConvolutionalNetwork::SaveWeights() {
  // Saves the calculated biases and weights (in both the fully-connected
  // layers and the (convolutional) kernels).
  std::ofstream save_file_stream(save_file_);
  save_file_stream << activation_function_in_fully_connected_layers_ << '\n' << s_shaped_vectors_ << '\n' << leaky_ReLU_in_convolutional_layers_ << '\n' << num_of_convolutional_and_pooling_layers_ << '\n' << num_of_feature_maps_ << '\n' << kernel_sizes_[0] << ' ' << kernel_sizes_[1] << '\n';
  for (unsigned i = 0; i < fully_connected_layer_sizes_.size(); ++i)
     save_file_stream << fully_connected_layer_sizes_[i] << ((i == (fully_connected_layer_sizes_.size()-1))? '\n' : ' ');
  for (unsigned i = 0; i < num_of_convolutional_and_pooling_layers_; ++i) {
    for (unsigned j = 0; j < num_of_feature_maps_; ++j) {
      for (unsigned k = 0; k < kernel_sizes_[0]; ++k) {
        for (unsigned l = 0; l < kernel_sizes_[1]; ++l)
          save_file_stream << kernels_[i][j][k][l] << ((k == (kernel_sizes_[0]-1) && l == (kernel_sizes_[1]-1))? '\n' : ' ');
      }
    }
  }
  for (unsigned i = 0; i < flattening_weights_.size(); ++i) {
    for (unsigned j = 0; j < flattening_weights_[i].size(); ++j) {
      for (unsigned k = 0; k < flattening_weights_[i][j].size(); ++k) {
        for (unsigned l = 0; l < flattening_weights_[i][j][k].size(); ++l)
          save_file_stream << flattening_weights_[i][j][k][l] << ((j == (flattening_weights_[i].size()-1) && k == (flattening_weights_[i][j].size()-1) && l == (flattening_weights_[i][j][k].size()-1))? '\n' : ' ');
      }
    }
  }
  for (unsigned i = 0; i < weights_.size(); ++i) {
    for (unsigned j = 0; j < weights_[i].size(); ++j) {
      for (unsigned k = 0; k < weights_[i][j].size(); ++k)
        save_file_stream << weights_[i][j][k] << ((j == (weights_[i].size()-1) && k == (weights_[i][j].size()-1))? '\n' : ' ');
    }
  }
  for (unsigned i = 0; i < biases_.size(); ++i) {
    for (unsigned j = 0; j < biases_[i].size(); ++j) {
      save_file_stream << biases_[i][j];
      if (i != (biases_.size()-1) && j != (biases_[i].size()-1))
        save_file_stream << ' ';
    }
  }
  std::cout << "(Weights and biases saved.)" << std::endl;
}

std::vector<unsigned> ConvolutionalNetwork::GetFullyConnectedLayerSizes(const std::string& load_file) {
  std::ifstream load_file_stream(load_file);
  std::string line, value;
  for (unsigned i = 0; i < 6; ++i)
    std::getline(load_file_stream, line);
  std::getline(load_file_stream, line);
  std::stringstream stream_fully_connected_layer_sizes(line);
  std::vector<unsigned> fully_connected_layer_sizes;
  while (getline(stream_fully_connected_layer_sizes, value, ' '))
    fully_connected_layer_sizes.push_back(std::stoi(value));
  return fully_connected_layer_sizes;
}

unsigned ConvolutionalNetwork::GetNumOfFeatureMaps(const std::string& load_file) {
  std::ifstream load_file_stream(load_file);
  std::string line, value;
  for (unsigned i = 0; i < 4; ++i)
    std::getline(load_file_stream, line);
  std::getline(load_file_stream, line);
  return std::stoi(line);
}

unsigned ConvolutionalNetwork::GetNumOfConvolutionalLayers(const std::string& load_file) {
  std::ifstream load_file_stream(load_file);
  std::string line, value;
  for (unsigned i = 0; i < 3; ++i)
    std::getline(load_file_stream, line);
  std::getline(load_file_stream, line);
  return std::stoi(line);
}

std::vector<unsigned> ConvolutionalNetwork::GetKernelSizes(const std::string& load_file) {
  std::ifstream load_file_stream(load_file);
  std::string line, value;
  for (unsigned i = 0; i < 5; ++i)
    std::getline(load_file_stream, line);
  std::getline(load_file_stream, line);
  std::stringstream stream_kernel_sizes(line);
  std::vector<unsigned> kernel_sizes(2);
  for (unsigned i = 0; i < 2; ++i) {
    getline(stream_kernel_sizes, value, ' ');
    kernel_sizes[i] = std::stoi(value);
  }
  return kernel_sizes;
}

bool ConvolutionalNetwork::GetActivationFunctionOfConvolutionalLayers(const std::string& load_file) {
  std::ifstream load_file_stream(load_file);
  std::string line, value;
  std::getline(load_file_stream, line);
  std::getline(load_file_stream, line);
  return std::stoi(line);
}

void ConvolutionalNetwork::OnDuty() {
  // Classifies the inputs using a network of pre-trained biases and weights.
  std::vector<std::vector<std::string>> classified_words(fully_connected_layer_sizes_[num_of_fully_connected_layers_-1]);
  std::vector<double>& output_layer = fully_connected_layers_[num_of_fully_connected_layers_-1];
  for (unsigned i = 0; i < train_size_; ++i) {
    Feedforward(i, false);
    classified_words[std::distance(output_layer.begin(), std::max_element(output_layer.begin(), output_layer.end()))].push_back(words_[0][i]);
  }
  SaveClassifications(classified_words);
}
