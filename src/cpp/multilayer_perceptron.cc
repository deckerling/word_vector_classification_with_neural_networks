// multilayer_perceptron.cc

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
#include <forward_list>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>

#include "mlp.h"

MultilayerPerceptron::MultilayerPerceptron(const std::vector<std::vector<double>>& train_input, const std::vector<std::vector<unsigned>>& train_output, const std::vector<std::vector<double>>& test_input, const std::vector<unsigned>& test_output, const std::vector<std::vector<std::string>>& words, const std::vector<unsigned>& layer_sizes, const unsigned epochs, const unsigned mini_batch_size, const double lambda, const double learning_rate, const std::string& save_file, const std::string& load_file, const unsigned verbose, const unsigned activation_function)
    : NeuralNetwork(train_output, test_output, words, epochs, mini_batch_size, lambda, learning_rate, activation_function, save_file, load_file),
      train_input_(train_input),
      test_input_(test_input),
      layer_sizes_(layer_sizes),
      number_of_layers_(layer_sizes.size()),
      number_of_neuron_layers_(number_of_layers_-1),
      verbose_(verbose) { // verbose == 0 => nothing more than the number of
                          // already completed epochs will be printed;
                          // verbose == 1 => the number of already completed
                          // epochs and the cost of every epoch will be printed;
                          // verbose == 2 => the number of already completed
                          // epochs, the cost, and the accuracy of every epoch
                          // will be printed
  std::cout << "- Done." << '\n' << "Initializing network..." << std::endl;
  biases_.resize(number_of_neuron_layers_);
  weights_.resize(number_of_neuron_layers_);
  activations_.resize(number_of_layers_);
  netinputs_.resize(number_of_neuron_layers_);
  nabla_w_.resize(number_of_neuron_layers_);
  (load_file_.empty())? InitializeBiasesAndWeights() : LoadBiasesAndWeights();
  std::cout << "- Done." << '\n' << "Selected hyper-parameters: " << '\n' << '\t' << "Number of layers (including in- and output layer): " << number_of_layers_ << '\n' << '\t' << "Layer sizes: ";
  for (auto it = layer_sizes_.begin(); it != layer_sizes_.end(); ++it)
    std::cout << *it << ((it == std::prev(layer_sizes_.end()))? '\n' : '/');
  std::cout << '\t' << "Activation function: ";
  switch (activation_function_in_fully_connected_layers_) {
   case 0:
    std::cout << "ReLU" << '\n';
    break;
   case 1:
    std::cout << "leaky ReLU" << '\n';
    break;
   case 2:
    std::cout << "Sigmoid" << '\n';
    break;
   default:
    std::cout << "tanh" << '\n';
  }
  std::cout << '\t' << "Epochs: " << epochs_ << '\n' << '\t' << "Mini-batch size: " << mini_batch_size_ << '\n' << '\t' << "Lambda: " << lambda_ << '\n' << '\t' << "Learning rate (eta): " << learning_rate_ << '\n' << "Initial test started..." << std::endl;
  if (test_size_ != 0) Evaluate(-1); // tests the initialized network before
                                     // the training has started
  std::cout << "Training started..." << std::endl;
  StochasticGradientDescent();
}

MultilayerPerceptron::MultilayerPerceptron(const std::vector<std::vector<double>>& input, const std::vector<std::vector<std::string>>& words, const std::string& save_directory, const std::string& load_file)
    : NeuralNetwork(input.size(), words, save_directory, load_file),
      train_input_(input),
      test_input_({{}}),
      layer_sizes_(GetLayerSizes(load_file)),
      number_of_layers_(layer_sizes_.size()),
      number_of_neuron_layers_(number_of_layers_-1),
      verbose_(0) {
  std::cout << "- Done." << '\n' << "Loading network..." << std::endl;
  biases_.resize(number_of_neuron_layers_);
  weights_.resize(number_of_neuron_layers_);
  activations_.resize(number_of_layers_);
  netinputs_.resize(number_of_neuron_layers_);
  nabla_w_.resize(number_of_neuron_layers_);
  LoadBiasesAndWeights();
  std::cout << "- Done." << '\n' << "Starting to classify the inputs..." << std::endl;
  OnDuty();
  std::cout << "- Done. Classifications have been saved." << '\n' << "Program terminated." << std::endl;
}

void MultilayerPerceptron::InitializeBiasesAndWeights() {
  // Initializes the biases and weights randomly.
  std::random_device ran_dev {};
  std::mt19937 generator {ran_dev()};
  std::normal_distribution <> distribution_biases {0, 1};
  auto GetRandomBias = [&distribution_biases, &generator](){return distribution_biases(generator);};
  for (unsigned i = 0; i < number_of_layers_; ++i) {
    if (i != number_of_neuron_layers_) {
      biases_[i].resize(layer_sizes_[i+1]);
      std::generate(biases_[i].begin(), biases_[i].end(), GetRandomBias);
      weights_[i].resize(layer_sizes_[i+1]); // the size of the first dimension
                                             // of each weight matrix is equal
                                             // to the number of neurons the
                                             // weights "lead to"
      std::normal_distribution <> distribution_weights {0, (1/std::sqrt(layer_sizes_[i]))};
      auto GetRandomWeight = [&distribution_weights, &generator]() {return distribution_weights(generator);};
      nabla_w_[i].resize(layer_sizes_[i+1]);
      for (unsigned j = 0; j < layer_sizes_[i+1]; ++j) {
        std::vector<double>& temp_weights = weights_[i][j];
        temp_weights.resize(layer_sizes_[i]); // the size of the second
                                              // dimension of each weight
                                              // matrix is equal to the number
                                              // of neurons the weights "start
                                              // from"
        std::generate(temp_weights.begin(), temp_weights.end(), GetRandomWeight);
        nabla_w_[i][j].resize(layer_sizes_[i]);
      }
      netinputs_[i].resize(layer_sizes_[i+1]);
    }
    activations_[i].resize(layer_sizes_[i]);
  }
}

void MultilayerPerceptron::LoadBiasesAndWeights() {
  // Loads biases and weights that were generated and saved during previous
  // trainings of the MLP.
  std::ifstream load_file_stream(load_file_);
  std::string line, value, values;
  std::getline(load_file_stream, line);
  std::getline(load_file_stream, line);
  std::stringstream stream_layer_sizes(line);
  std::vector<unsigned> layer_sizes_to_check;
  while (getline(stream_layer_sizes, value, ' '))
    layer_sizes_to_check.push_back(std::stoi(value));
  if (!(layer_sizes_to_check == layer_sizes_)) { // checks if the chosen file
                                                 // matches the network's
                                                 // design.
    std::cout << "ERROR: The weights and biases found in the file you wanted to load do not match the layer sizes you have chosen!" << '\n' << "The weights and biases will be initialized randomly (again)..." << std::endl;
    InitializeBiasesAndWeights();
    return;
  }
  std::getline(load_file_stream, line);
  std::stringstream stream_biases(line);
  for (unsigned i = 0; i < number_of_layers_; ++i) {
    if (i != number_of_neuron_layers_) {
      getline(stream_biases, values, ',');
      std::stringstream stream_biases_per_layer(values);
      while (getline(stream_biases_per_layer, value, ' '))
        biases_[i].push_back(atof(value.c_str()));
      netinputs_[i] = std::vector<double>(layer_sizes_[i+1]);
      nabla_w_[i] = std::vector<std::vector<double>>(layer_sizes_[i+1]);
      for (auto& nw_i : nabla_w_[i])
        nw_i = std::vector<double>(layer_sizes_[i]);
    }
    activations_[i] = std::vector<double>(layer_sizes_[i]);
  }
  std::getline(load_file_stream, line);
  std::stringstream stream_weights(line);
  for (unsigned i = 0; i < number_of_neuron_layers_; ++i) {
    weights_[i].resize(layer_sizes_[i+1]);
    getline(stream_weights, values, ',');
    std::stringstream stream_weights_per_layer(values);
    for (unsigned j = 0; j < layer_sizes_[i+1]; ++j) {
      std::vector<double>& temp_weights = weights_[i][j];
      temp_weights.resize(layer_sizes_[i]);
      for (auto& weight : temp_weights) {
        getline(stream_weights_per_layer, value, ' ');
        weight = atof(value.c_str());
      }
    }
  }
}

std::vector<unsigned> MultilayerPerceptron::GetLayerSizes(std::string load_file) {
  // If the network works on duty the layer sizes will be read from the
  // "load_file".
  std::ifstream load_file_stream(load_file);
  std::string line, value;
  std::getline(load_file_stream, line);
  std::getline(load_file_stream, line);
  std::stringstream stream_layer_sizes(line);
  std::vector<unsigned> layer_sizes;
  while (getline(stream_layer_sizes, value, ' '))
    layer_sizes.push_back(std::stoi(value));
  return layer_sizes;
}

void MultilayerPerceptron::StochasticGradientDescent() {
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

void MultilayerPerceptron::UpdateMiniBatches(const unsigned mini_batch, const std::vector<unsigned>& mini_batch_indices, const bool last_mini_batch) {
  // Starts the backpropagation for each mini batch and calculates the error
  // and the new biases and weights.
  const unsigned& current_mini_batch_size = (last_mini_batch && mini_batch_size_remainder_ != 0)? mini_batch_size_remainder_ : mini_batch_size_;
  std::vector<std::vector<std::vector<double>>> all_nabla_b(current_mini_batch_size, std::vector<std::vector<double>>(number_of_neuron_layers_));
  std::vector<std::vector<std::vector<std::vector<double>>>> all_nabla_w(current_mini_batch_size, std::vector<std::vector<std::vector<double>>>(number_of_neuron_layers_));
  unsigned i = 0;
  for (unsigned j = (mini_batch*mini_batch_size_); j < (mini_batch*mini_batch_size_+current_mini_batch_size); ++j) {
    Backpropagate(mini_batch_indices[j]);
    all_nabla_b[i] = nabla_b_;
    all_nabla_w[i] = nabla_w_;
    i++;
  }
  // Sums up the nabla values for biases and weights of the mini batch.
  for (unsigned i = 1; i < current_mini_batch_size; ++i) {
    for (unsigned j = 0; j < number_of_neuron_layers_; ++j) {
      std::transform(all_nabla_b[0][j].begin(), all_nabla_b[0][j].end(), all_nabla_b[i][j].begin(), all_nabla_b[0][j].begin(), std::plus<double>());
      const std::vector<std::vector<double>>& current_nabla_w = all_nabla_w[i][j];
      std::vector<std::vector<double>>& current_nabla_w_sum = all_nabla_w[0][j];
      for (unsigned k = 0; k < layer_sizes_[j+1]; ++k)
        std::transform(current_nabla_w_sum[k].begin(), current_nabla_w_sum[k].end(), current_nabla_w[k].begin(), current_nabla_w_sum[k].begin(), std::plus<double>());
    }
  }
  // Updates the biases and weights:
  //   biases_ = biases_ - (learning_rate_/current_mini_batch_size) * all_nabla_b_sum
  //   weights_ = weights_ * (1-learning_rate_*(lambda_/train_size_)) - learning_rate_/current_mini_batch_size * all_nabla_w_sum;
  const double scalar0 = learning_rate_/current_mini_batch_size;
  const double scalar1 = 1-learning_rate_*(lambda_/train_size_); // used for regularization
  for (unsigned i = 0; i < number_of_neuron_layers_; ++i) {
    std::transform(all_nabla_b[0][i].begin(), all_nabla_b[0][i].end(), all_nabla_b[0][i].begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, scalar0));
    std::transform(biases_[i].begin(), biases_[i].end(), all_nabla_b[0][i].begin(), biases_[i].begin(), std::minus<double>());
    std::vector<std::vector<double>>& nabla_w_sum = all_nabla_w[0][i];
    std::vector<std::vector<double>>& current_weights = weights_[i];
    for (unsigned j = 0; j < layer_sizes_[i+1]; ++j) {
      std::transform(nabla_w_sum[j].begin(), nabla_w_sum[j].end(), nabla_w_sum[j].begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, scalar0));
      std::transform(current_weights[j].begin(), current_weights[j].end(), current_weights[j].begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, scalar1));
      std::transform(current_weights[j].begin(), current_weights[j].end(), nabla_w_sum[j].begin(), current_weights[j].begin(), std::minus<double>());
    }
  }
}

void MultilayerPerceptron::Backpropagate(const unsigned train_data_index) {
  // Calculates the errors.
  Feedforward(train_data_index, false);
  std::vector<std::vector<double>> deltas(number_of_neuron_layers_);
  for (unsigned i = 0; i < number_of_neuron_layers_; ++i)
    deltas[i].resize(layer_sizes_[i+1]);
  std::transform(activations_[number_of_neuron_layers_].begin(), activations_[number_of_neuron_layers_].end(), train_output_[train_data_index].begin(), deltas[number_of_neuron_layers_-1].begin(), std::minus<double>());
  for (unsigned i = (number_of_neuron_layers_-1); i > 0; --i) {
  // Calculates the delta values moving backwards (i.e. starting with the
  // output error).
    std::fill(deltas[i-1].begin(), deltas[i-1].end(), 0);
    for (unsigned j = 0; j < layer_sizes_[i]; ++j) {
      double& current_delta = deltas[i-1][j];
      for (unsigned k = 0; k < layer_sizes_[i+1]; ++k)
        current_delta += deltas[i][k]*weights_[i][k][j];
      current_delta *= DerivatedActivationFunctionInFullyConnectedLayers(netinputs_[i-1][j]);
    }
  }
  nabla_b_ = deltas;
  for (unsigned i = 0; i < number_of_neuron_layers_; ++i) {
    std::vector<std::vector<double>>& nw_i = nabla_w_[i];
    for (unsigned j = 0; j < layer_sizes_[i+1]; ++j) {
      const double& current_delta = deltas[i][j];
      for (unsigned k = 0; k < layer_sizes_[i]; ++k)
        nw_i[j][k] = current_delta*activations_[i][k];
    }
  }
}

void MultilayerPerceptron::Feedforward(const unsigned data_index, const bool test) {
  // Finds the networks output for a given input.
  activations_[0] = (test)? test_input_[data_index] : train_input_[data_index];
  netinputs_ = biases_;
  for (unsigned i = 0; i < number_of_neuron_layers_; ++i) {
    for (unsigned j = 0; j < layer_sizes_[i+1]; ++j) {
      const std::vector<double>& current_weights = weights_[i][j];
      double& current_netinput = netinputs_[i][j];
      for (unsigned k = 0; k < layer_sizes_[i]; ++k)
        current_netinput += current_weights[k]*activations_[i][k];
    }
    if (i != (number_of_neuron_layers_-1)) {
      for (unsigned j = 0; j < layer_sizes_[i+1]; ++j)
        activations_[i+1][j] = ActivationFunctionInFullyConnectedLayers(netinputs_[i][j]);
    } else {
      std::vector<double>& output_netinputs = netinputs_[number_of_neuron_layers_-1];
      if (activation_function_in_fully_connected_layers_ != 2) { // softmax function will be used in the output layer
        // Softmax:
        double sum = 0;
        for (auto& netinput : output_netinputs)
          sum += std::exp(netinput);
        for(unsigned i = 0; i < layer_sizes_[number_of_neuron_layers_]; ++i)
          activations_[number_of_neuron_layers_][i] = std::exp(output_netinputs[i])/sum;
      } else { // sigmoid function will be used in the output layer
        // Sigmoid:
        for (unsigned i = 0; i < layer_sizes_[number_of_neuron_layers_]; ++i)
          activations_[number_of_neuron_layers_][i] = ActivationFunctionInFullyConnectedLayers(output_netinputs[i]);
      }
    }
  }
}

void MultilayerPerceptron::Evaluate(const int epoch) {
  // Checks how many of the inputs of the test data get classified correctly by
  // the network with the weights and biases currently used.
  std::cout << (epoch+1) << ". epoch completed.";
  unsigned correct = 0, wrong = 0;
  double cost = 0, sum_of_squared_weights = 0;
  std::forward_list<std::string> wrongly_classified_words;
  for (unsigned i = 0; i < test_size_; ++i) {
    Feedforward(i, true);
    cost += CalculateCost(test_output_[i])/test_size_;
    if (verbose_ == 2) {
      std::vector<double>& output_layer = activations_[number_of_neuron_layers_];
      if (test_output_[i] == std::distance(output_layer.begin(), std::max_element(output_layer.begin(), output_layer.end())))
        correct++; // if the desired output is equal to the output neuron with
                   // the highest activation level
      else if (epoch == ((int) epochs_-1) && wrong < 100) {
        wrongly_classified_words.push_front(words_[1][i]);
        wrong++;
      }
    }
  }
  for (unsigned i = 0; i < number_of_neuron_layers_; ++i) {
    for (unsigned j = 0; j < layer_sizes_[i+1]; ++j) {
      for (auto& weight : weights_[i][j])
        sum_of_squared_weights += pow(weight, 2);
    }
  }
  cost += 0.5*(lambda_/test_size_)*sum_of_squared_weights;
  if (verbose_ != 0)
    std::cout << '\n' << '\t' << "Cost: " << cost;
  if (verbose_ == 2)
    std::cout << '\n' << '\t' << "Correct results: " << correct << '/' << test_size_ << " (accuracy: " << ((double) correct/(test_size_/100)) << "%)";
  std::cout << std::endl;
  if (epoch == -1 && !save_file_.empty())
    lowest_cost_ = cost;
  else if (!save_file_.empty() && cost < lowest_cost_) { // only the biases and
                                                         // weights that lead
                                                         // to the best result
                                                         // will be saved
    lowest_cost_ = cost;
    SaveWeights();
  }
  if (epoch == ((int) epochs_-1) && wrong != 0) {
    std::cout << "Words that were classified wrongly: " << '\n'; // shows not more than 100 wrongly classified words
    for (auto& wrong_word : wrongly_classified_words)
      std::cout << ' ' << wrong_word;
    std::cout << '\n';
  }
}

double MultilayerPerceptron::CalculateCost(unsigned current_test_output) {
  double sum = 0;
  for (unsigned i = 0; i < layer_sizes_[number_of_neuron_layers_]; ++i) {
    if (!(current_test_output == i && activations_[number_of_neuron_layers_][i] == 1))
      sum += (-((current_test_output==i)? 1 : 0)*log(activations_[number_of_neuron_layers_][i])-(1-((current_test_output==i)? 1 : 0))*log(1-activations_[number_of_neuron_layers_][i]));
  }
  return sum;
}

void MultilayerPerceptron::SaveWeights() {
  // Saves the current biases and weights:
  //  In the first line of the file the layer sizes will be saved (separated
  //  with a whitespace); in the second line the biases will be saved with all
  //  values separated with a whitespace and the biases of different layers
  //  separated with a comma - in a similar way the weights will be saved (i.e.
  //  the weights of each layer will be saved in a "one dimensional" way!).
  std::ofstream save_file_stream(save_file_);
  save_file_stream << activation_function_in_fully_connected_layers_ << '\n';
  for (auto it = layer_sizes_.begin(); it != layer_sizes_.end(); ++it)
    save_file_stream << *it << ((it == std::prev(layer_sizes_.end()))? '\n' : ' ');
  for (unsigned i = 0; i < number_of_neuron_layers_; ++i) {
    for (unsigned j = 0; j < layer_sizes_[i+1]; ++j)
      save_file_stream << biases_[i][j] << ((j == (layer_sizes_[i+1]-1))? "" : " ");
    save_file_stream << ((i == (number_of_neuron_layers_-1))? '\n' : ',');
  }
  for (unsigned i = 0; i < number_of_neuron_layers_; ++i) {
    for (unsigned j = 0; j < layer_sizes_[i+1]; ++j) {
      const std::vector<double>& temp_weights = weights_[i][j];
      for (unsigned k = 0; k < layer_sizes_[i]; ++k)
        save_file_stream << temp_weights[k] << ((k == (layer_sizes_[i]-1))? "" : " ");
      save_file_stream << ((j == (layer_sizes_[i+1]-1))? "" : " ");
    }
    save_file_stream << ((i == (number_of_neuron_layers_-1))? "" : ",");
  }
  std::cout << "(Biases and weights saved.)" << std::endl;
}

void MultilayerPerceptron::OnDuty() {
  // Classifies the inputs using a network of pre-trained biases and weights.
  std::vector<std::vector<std::string>> classified_words(layer_sizes_[number_of_neuron_layers_]);
  std::vector<double>& output_layer = activations_[number_of_neuron_layers_];
  for (unsigned i = 0; i < train_size_; ++i) {
    Feedforward(i, false);
    classified_words[std::distance(output_layer.begin(), std::max_element(output_layer.begin(), output_layer.end()))].push_back(words_[0][i]);
  }
  SaveClassifications(classified_words);
}
