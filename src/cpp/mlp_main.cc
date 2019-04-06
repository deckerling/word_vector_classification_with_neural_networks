// mlp_main.cc

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

#include <cstring>
#include <sstream>

#include "data_loader.h"
#include "mlp.h"

void PrintErrorAndStyleOfUsage(const std::string& error_message) {
  std::cout << error_message << '\n';
  std::cout << "Style of usage for training:\n\t.\\mlp4wordvec -train_in [file containing input training data] -train_out [file containing output training data] -test_in [file containing input test data] -test_out [file containing output test data] -load_file [file containing pre-trained biases and weights] -save_file [file to save biases and weights in] -layer_sizes [number of neurons per layer (e.g. \"300/40/10\")] -epochs [number of training epochs] -mini_batch [mini batch size] -eta [learning rate] -lambda [lambda (used for regularization)], -verbose ['0' to show nothing more than the number of already completed epochs, '1' to show the cost, and '2' to show the accuracy as well], -activation [activation function (\"relu\" (or '0'), \"leaky_relu\" (or '1'), \"sigmoid\" (or '2'), or \"tanh\" (or '3')) - if pre-trained biases and weights are used the activation function will be taken from their file as well]" << '\n';
  std::cout << "Default settings for training:" << '\n' << '\t' << "-train_in [NO DEFAULT!]" << '\n' << '\t' << "-train_out [NO DEFAULT!]" << '\n' << '\t' << "-test_in [NO DEFAULT!]" << '\n' << '\t' << "-test_out [NO DEFAULT!]" << '\n' << '\t' << "-load_file [NO DEFAULT!]" << '\n' << '\t' << "-save_file [NO DEFAULT!]" << '\n' << '\t' << "-layer_sizes 0/0 [a net without hidden layers will be created after analyzing the training data]" << '\n' << '\t' << "-epochs 25" << '\n' << '\t' << "-mini_batch [a default value will be calculated with respect to the size of the training data]" << '\n' << '\t' << "-eta 0.7" << '\n' << '\t' << "-lambda 0 [no regularization will take place]" << '\n' << '\t' << "-verbose 1" << '\n' << '\t' << "-activation tanh" << '\n';
  std::cout << "NOTE: The arguments \"-train_in\" and \"-train_out\" are not optional for the training!" << '\n';
  std::cout << "Style of usage for actual usage of a pre-trained network:\n\t.\\mlp4wordvec -input [file containing input data that should be classified] -load_file [file containing pre-trained biases and weights] -save-dir [existing directory to save the classification results in]" << '\n';
  std::cout << "Default settings for actual usage of a pre-trained network:" << '\n' << '\t' << "-input [NO DEFAULT!]" << '\n' << '\t' << "-load_file [NO DEFAULT!]" << '\n' << '\t' << "-save_dir [the default directory is the same as this program is saved in]" << '\n';
  std::cout << "NOTE: The arguments \"-input\" and \"-load_file\" are not optional for the actual usage of the network!" << '\n';
  std::cout << "Example usage for training:\n\t.\\mlp4wordvec -train_in train_input.txt -train_out train_output.txt" << '\n' << "Example usage for actual usage:\n\t.\\mlp4wordvec -input input.txt -load_file my_weights_and_biases.csv" << '\n' << "Program terminated." << std::endl;
}

unsigned FindArgument(const int argc, char* argv[], const std::string& argument_to_search_for) {
  // Checks whether a certain argument is given or not and returns its index
  // (if it was found).
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == argument_to_search_for)
      return i;
  }
  return 0;
}

int main(int argc, char* argv[]) {
  // Searches for the arguments.
  if (argc < 2) {
    PrintErrorAndStyleOfUsage("FATAL ERROR: Not enough arguments provided!");
    return -1;
  }
  unsigned i, j;
  if ((i = FindArgument(argc, argv, "-input")) > 0 && (j = FindArgument(argc, argv, "-load_file")) > 0) { // if a pre-trained network gets actually used
    std::string input_file = argv[i+1];
    const std::string load_file = argv[j+1];
    if (!FileIsValid(load_file)) {
      std::cout << "FATAL ERROR: Weights couldn't be loaded!" << '\n' << "Program terminated." << std::endl;
      return -1;
    }
    const std::string save_directory = ((i = FindArgument(argc, argv, "-save_dir")) > 0)? argv[i+1] : "";
    // Loads data...
    Loader loader(input_file);
    const std::vector<std::vector<double>> input = loader.LoadTrainInput();
    // Initializes network and starts classifying...
    MultilayerPerceptron network(input, loader.GetWords(), save_directory, load_file);
    return 0;
  } else { // if the network shall be trained
    std::vector<std::string> data_files(4);
    if ((i = FindArgument(argc, argv, "-train_in")) > 0 && (j = FindArgument(argc, argv, "-train_out")) > 0) {
      data_files[0] = argv[i+1];
      data_files[1] = argv[j+1];
    } else {
      PrintErrorAndStyleOfUsage("Training impossible: No training data provided!");
      return -1;
    }
    if ((i = FindArgument(argc, argv, "-test_in")) > 0)
      data_files[2] = argv[i+1];
    if ((i = FindArgument(argc, argv, "-test_out")) > 0)
      data_files[3] = argv[i+1];
    std::string load_file = ((i = FindArgument(argc, argv, "-load_file")) > 0)? argv[i+1] : "";
    const std::string save_file = ((i = FindArgument(argc, argv, "-save_file")) > 0)? argv[i+1] : "";
    std::vector<unsigned> layer_sizes;
    if ((i = FindArgument(argc, argv, "-layer_sizes")) > 0) {
      std::stringstream stream(argv[i+1]);
      std::string layer;
      while (getline(stream, layer, '/'))
        layer_sizes.push_back(std::stoi(layer));
    } else
      layer_sizes = {0, 0};
    const unsigned epochs = ((i = FindArgument(argc, argv, "-epochs")) > 0)? std::stoi(argv[i+1]) : 25;
    const unsigned mini_batch_size = ((i = FindArgument(argc, argv, "-mini_batch")) > 0)? std::stoi(argv[i+1]) : 0;
    const double lambda = ((i = FindArgument(argc, argv, "-lambda")) > 0)? atof(argv[i+1]) : 0;
    const double learning_rate = ((i = FindArgument(argc, argv, "-eta")) > 0)? atof(argv[i+1]) : 0.7;
    const unsigned verbose = ((i = FindArgument(argc, argv, "-verbose")) > 0)? std::stoi(argv[i+1]) : 1;
    unsigned activation_function = 3; // the default activation function is tanh
    if ((i = FindArgument(argc, argv, "-activation")) > 0) {
      if (strcmp(argv[i+1], "0") == 0 || strcmp(argv[i+1], "relu") == 0)
        activation_function = 0;
      else if (strcmp(argv[i+1], "1") == 0 || strcmp(argv[i+1], "leaky_relu") == 0)
        activation_function = 1;
      else if (strcmp(argv[i+1], "2") == 0 || strcmp(argv[i+1], "sigmoid") == 0)
        activation_function = 2;
    }
    // Checks the files...
    for (auto& file_path : data_files) {
      if(!FileIsValid(file_path)) {
        std::cout << "FATAL ERROR: Data couldn't be loaded!" << '\n' << "Program terminated." << std::endl;
        return -1;
      }
    }
    if (!load_file.empty() && !FileIsValid(load_file)) {
      load_file = "";
      std::cout << "ERROR: The weights and biases will be initialized randomly (again)..." << std::endl;
    }
    // Loads data...
    Loader loader(data_files, layer_sizes);
    const std::vector<std::vector<unsigned>> train_output = loader.LoadTrainOutput();
    const std::vector<std::vector<double>> train_input = loader.LoadTrainInput();
    const std::vector<std::vector<double>> test_input = loader.LoadTestInput();
    // Initializes network and starts training...
    MultilayerPerceptron network(train_input, train_output, test_input, loader.LoadTestOutput(), loader.GetWords(), loader.GetFinalLayerSizes(), epochs, ((mini_batch_size != 0)? mini_batch_size : ((((unsigned) train_output.size()/20+1) < 100)? ((unsigned) train_output.size()/20+1) : 100)), lambda, learning_rate, save_file, load_file, verbose, activation_function);
  }
  return 0;
}
