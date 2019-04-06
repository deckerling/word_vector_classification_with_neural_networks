// neural_network.cc

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
#include <random>

#include "neural_network.h"

NeuralNetwork::NeuralNetwork(const std::vector<std::vector<unsigned>>& train_output, const std::vector<unsigned>& test_output, const std::vector<std::vector<std::string>>& words, const unsigned epochs, const unsigned mini_batch_size, const double lambda, const double learning_rate, const unsigned activation_function_in_fully_connected_layers, const std::string& save_file, const std::string& load_file)
    : train_output_(train_output), // used as "desired" output
      test_output_(test_output),
      train_size_(train_output.size()),
      test_size_(test_output.size()),
      words_(words),
      epochs_(epochs),
      mini_batch_size_(CheckMiniBatchSize(mini_batch_size)),
      lambda_(lambda),
      learning_rate_(learning_rate),
      mini_batch_size_remainder_(train_size_%mini_batch_size_),
      // Activation function:
      //   activation_function == 0 => ReLU;
      //   activation_function == 1 => leaky ReLU;
      //   activation_function == 2 => Sigmoid;
      //   else => tanh
      activation_function_in_fully_connected_layers_((!load_file.empty())? GetActivationFunctionOfFullyConnectedLayers(load_file, activation_function_in_fully_connected_layers) : activation_function_in_fully_connected_layers),
      save_file_(save_file),
      load_file_(load_file) {}

NeuralNetwork::NeuralNetwork(const unsigned input_size, const std::vector<std::vector<std::string>>& words, const std::string& save_directory, const std::string& load_file)
    : train_output_({{}}),
      test_output_({}),
      train_size_(input_size),
      test_size_(0),
      words_(words),
      epochs_(0),
      mini_batch_size_(0),
      lambda_(0),
      learning_rate_(0),
      mini_batch_size_remainder_(0),
      activation_function_in_fully_connected_layers_(GetActivationFunctionOfFullyConnectedLayers(load_file, 0)),
      save_file_(save_directory),
      load_file_(load_file) {}

unsigned NeuralNetwork::GetActivationFunctionOfFullyConnectedLayers(const std::string& load_file, const unsigned activation_function_in_fully_connected_layers) {
  // Checks if the given "activation_function_in_fully_connected_layers" is
  // equal to the "activation_function_in_fully_connected_layers" found in the
  // "load_file" - if not the latter will be returned.
  std::ifstream load_file_stream(load_file);
  std::string line, value;
  std::getline(load_file_stream, line);
  std::cout << "Activation function adjusted." << std::endl;
  return (std::stoi(line) != ((int) activation_function_in_fully_connected_layers))? std::stoi(line) : activation_function_in_fully_connected_layers;
}

void NeuralNetwork::SaveClassifications(std::vector<std::vector<std::string>>& classified_words) {
  // Saves the classified words in txt-files with every file representing a
  // particular class.
  std::string save_dir = (save_file_.empty())? save_file_ : save_file_+"/";
  for (unsigned i = 0; i < classified_words.size(); ++i) {
    std::ofstream classification_file(save_dir+"class"+std::to_string(i)+".txt");
    for (auto& word : classified_words[i])
      classification_file << word << '\n';
  }
}
