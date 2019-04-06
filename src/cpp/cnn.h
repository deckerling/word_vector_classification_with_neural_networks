// cnn.h

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

#ifndef WORD_VECTOR_CLASSIFICATION_WITH_NEURAL_NETWORKS_SRC_CPP_CNN_H_
#define WORD_VECTOR_CLASSIFICATION_WITH_NEURAL_NETWORKS_SRC_CPP_CNN_H_

#include "neural_network.h"

class ConvolutionalNetwork : public NeuralNetwork {
 public:
  // Constructor for a network to be trained:
  ConvolutionalNetwork(const std::vector<std::vector<std::vector<double>>>& train_input,
      const std::vector<std::vector<unsigned>>& train_output,
      const std::vector<std::vector<std::vector<double>>>& test_input,
      const std::vector<unsigned>& test_output,
      const std::vector<std::vector<std::string>>& words,
      const std::vector<unsigned>& fully_connected_layer_sizes,
      const unsigned num_of_feature_maps,
      const unsigned num_of_convolutional_and_pooling_layers,
      const std::vector<unsigned>& kernel_sizes,
      const unsigned pooling,
      const bool leaky_ReLU_in_convolutional_layers,
      const unsigned activation_function_in_fully_connected_layers,
      const unsigned epochs,
      const unsigned mini_batch_size,
      const double lambda,
      const double learning_rate,
      const std::string& save_file,
      const std::string& load_file,
      const bool s_shaped_vectors);
  // Constructor for a pre-trained network that should do its duty:
  ConvolutionalNetwork(const std::vector<std::vector<std::vector<double>>>& input,
      const std::vector<std::vector<std::string>>& words,
      const std::string& save_directory,
      const std::string& load_file,
      const unsigned pooling);

 private:
  // Data:
  const std::vector<std::vector<std::vector<double>>> train_input_, test_input_;
  // Hyper-parameters:
  const std::vector<unsigned> fully_connected_layer_sizes_;
  const unsigned num_of_feature_maps_, num_of_convolutional_and_pooling_layers_, num_of_fully_connected_layers_;
  const std::vector<unsigned> kernel_sizes_;
  const std::vector<bool> even_kernel_sizes_;
  const unsigned pooling_;
  const bool leaky_ReLU_in_convolutional_layers_;

  bool s_shaped_vectors_;
  unsigned best_test_result = 0;
  std::vector<std::vector<std::vector<std::vector<double>>>> convolutional_layers_, convolutional_netinputs_;
  std::vector<std::vector<double>> fully_connected_layers_;
  std::vector<std::vector<std::vector<std::vector<double>>>> kernels_, nabla_kernels_;
  std::vector<std::vector<double>> biases_, nabla_b_, fully_connected_netinputs_;
  std::vector<std::vector<std::vector<std::vector<double>>>> flattening_weights_, flattening_nabla_w_;
  std::vector<std::vector<std::vector<double>>> weights_, nabla_w_;

  void InitializeBiasesWeightsKernels(const bool s_shaped_vectors);
  void LoadBiasesWeightsKernels();
  void StochasticGradientDescent();
  void UpdateMiniBatches(const unsigned mini_batch, const std::vector<unsigned>& mini_batch_indices, const bool last_mini_batch);
  void Backpropagate(const unsigned train_data_index);
  void Feedforward(const unsigned train_data_index, const bool test);
  void Pooling(const unsigned i);
  double ActivationFunctionInConvolutionalLayers(const double netinput) {
    return (leaky_ReLU_in_convolutional_layers_)? LeakyReLU(netinput) : ReLU(netinput);
  }
  double DerivatedActivationFunctionInConvolutionalLayers(const double netinput) {
    return (leaky_ReLU_in_convolutional_layers_)? DerivatedLeakyReLU(netinput) : DerivatedReLU(netinput);
  }
  void Evaluate(const int epoch);
  void SaveWeights();

  std::vector<unsigned> GetFullyConnectedLayerSizes(const std::string& load_file);
  unsigned GetNumOfFeatureMaps(const std::string& load_file);
  unsigned GetNumOfConvolutionalLayers(const std::string& load_file);
  std::vector<unsigned> GetKernelSizes(const std::string& load_file);
  bool GetActivationFunctionOfConvolutionalLayers(const std::string& load_file);

  void OnDuty();
};

#endif // WORD_VECTOR_CLASSIFICATION_WITH_NEURAL_NETWORKS_SRC_CPP_CNN_H_
