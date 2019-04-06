// mlp.h

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

#ifndef WORD_VECTOR_CLASSIFICATION_WITH_NEURAL_NETWORKS_SRC_CPP_MLP_H_
#define WORD_VECTOR_CLASSIFICATION_WITH_NEURAL_NETWORKS_SRC_CPP_MLP_H_

#include "neural_network.h"

class MultilayerPerceptron : public NeuralNetwork {
 public:
  // Constructor for a network to be trained:
  MultilayerPerceptron(const std::vector<std::vector<double>>& train_input,
      const std::vector<std::vector<unsigned>>& train_output,
      const std::vector<std::vector<double>>& test_input,
      const std::vector<unsigned>& test_output,
      const std::vector<std::vector<std::string>>& words,
      const std::vector<unsigned>& layer_sizes,
      const unsigned epochs,
      const unsigned mini_batch_size,
      const double lambda,
      const double learning_rate,
      const std::string& save_file,
      const std::string& load_file,
      const unsigned verbose,
      const unsigned activation_function);
  // Constructor for a pre-trained network that should do its duty:
  MultilayerPerceptron(const std::vector<std::vector<double>>& input,
      const std::vector<std::vector<std::string>>& words,
      const std::string& save_directory,
      const std::string& load_file);

 private:
  // Data:
  const std::vector<std::vector<double>> train_input_, test_input_;
  // Hyper-parameters:
  const std::vector<unsigned> layer_sizes_;
  const unsigned number_of_layers_, number_of_neuron_layers_;
  const unsigned verbose_;

  double lowest_cost_;
  std::vector<std::vector<double>> biases_, activations_, nabla_b_, netinputs_;
  std::vector<std::vector<std::vector<double>>> weights_, nabla_w_;

  void InitializeBiasesAndWeights();
  void LoadBiasesAndWeights();
  std::vector<unsigned> GetLayerSizes(std::string load_file);

  void StochasticGradientDescent();
  void UpdateMiniBatches(const unsigned mini_batch, const std::vector<unsigned>& mini_batch_indices, const bool last_mini_batch);
  void Backpropagate(const unsigned train_data_index);
  void Feedforward(const unsigned train_data_index, const bool test);
  void Evaluate(const int epoch);
  double CalculateCost(unsigned current_test_output);
  void SaveWeights();

  void OnDuty();
};

#endif // WORD_VECTOR_CLASSIFICATION_WITH_NEURAL_NETWORKS_SRC_CPP_MLP_H_
