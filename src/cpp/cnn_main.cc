// cnn_main.cc

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

#include "cnn.h"
#include "data_loader.h"

void PrintErrorAndStyleOfUsage(const std::string& error_message) {
  std::cout << error_message << '\n';
  std::cout << "Style of usage for training:\n\t.\\cnn4wordvec -train_in [file containing input training data] -train_out [file containing output training data] -test_in [file containing input test data] -test_out [file containing output test data] -load_file [file containing pre-trained biases and weights] -save_file [file to save biases and weights in] -layer_sizes [number of neurons per fully-connected layer (e.g. \"100/30/10\")] -conv_layers [number of convolutional and pooling layers] -filters [number of filters] -conv_act [activation function in the convolutional layers (\"leaky_relu\" or ('1') or \"relu\")] -fully-connect_act [activation function in the fully-connected layers (\"relu\" (or '0'), \"leaky_relu\" (or '1'), \"sigmoid\" (or '2'), or \"tanh\" (or '3'))] -pooling [pooling function (\"max\" (or '0'), \"L2\" (or '1'), or \"average\" (or '2'))] -kernel_sizes [size of the two-dimensional kernel (e.g. \"5/5\")] -s_shaped [if \"true\" the word vectors will be loaded in s-shape to perform their transformation from one- into two-dimensional] -epochs [number of training epochs] -mini_batch [mini_batch_size] -lambda [lambda (used for regularization)] -eta [learning_rate]" << '\n';
  std::cout << "Default settings for training:" << '\n' << '\t' << "-train_in [NO DEFAULT!]" << '\n' << '\t' << "-train_out [NO DEFAULT!]" << '\n' << '\t' << "-test_in [NO DEFAULT!]" << '\n' << '\t' << "-test_out [NO DEFAULT!]" << '\n' << '\t' <<"-load_file [NO DEFAULT!]" << '\n' << '\t' <<"-save_file [NO DEFAULT!]" << '\n' << '\t' <<"-layer_sizes 0/ [after analyzing the training data only an output layer will be created]" << '\n' << '\t' <<"-conv_layers 1" << '\n' << '\t' <<"-filters 8" << '\n' << '\t' <<"-conv_act 1 [ReLU]" << '\n' << '\t' <<"-fully-connect_act 3 [tanh]" << '\n' << '\t' <<"-pooling 0 [max pooling]" << '\n' << '\t' <<"-kernel_sizes 3/3" << '\n' << '\t' <<"-s_shaped 1" << '\n' << '\t' <<"-epochs 40" << '\n' << '\t' <<"-mini_batch [a default value will be calculated with respect to the size of the training data]" << '\n' << '\t' <<"-lambda 0 [no regularization will take place]" << '\n' << '\t' << "-eta 0.1" << '\n';
  std::cout << "NOTE: The arguments \"-train_in\" and \"-train_out\" are not optional for the training!" << '\n';
  std::cout << "Style of usage for actual usage of a pre-trained network:\n\t.\\cnn4wordvec -input [file containing input data that should be classified] -load_file [file containing pre-trained biases and weights] -save-dir [existing directory to save the classification results in] –pooling [pooling function (\"max\" (or '0'), \"L2\" (or '1'), or \"average\" (or '2')]" << '\n';
  std::cout << "Default settings for actual usage of a pre-trained network:" << '\n' << '\t' << "-input [NO DEFAULT!]" << '\n' << '\t' << "-load_file [NO DEFAULT!]" << '\n' << '\t' << "-save_dir [the default directory is the same as this program is saved in]" << '\n';
  std::cout << "NOTE: The arguments \"-input\" and \"-load_file\" are not optional for the actual usage of the network!" << '\n';
  std::cout << "Example usage for training:\n\t.\\cnn4wordvec -train_in train_input.txt -train_out train_output.txt" << '\n' << "Example usage for actual usage:\n\t.\\cnn4wordvec -input input.txt -load_file my_weights_and_biases.csv –pooling max" << '\n' << "Program terminated." << std::endl;
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
    unsigned pooling = 0; // 0 = max pooling
    if ((i = FindArgument(argc, argv, "-pooling")) > 0) {
      if (strcmp(argv[i+1], "1") == 0 || strcmp(argv[i+1], "L2") || strcmp(argv[i+1], "l2") == 0)
        pooling = 1;
      else if (strcmp(argv[i+1], "2") == 0 || strcmp(argv[i+1], "average") == 0)
        pooling = 2;
    }
    // Loads data...
    Loader loader(input_file, load_file);
    const std::vector<std::vector<std::vector<double>>> input = loader.ReshapeData(loader.LoadTrainInput());
    // Initializes network and starts classifying...
    ConvolutionalNetwork network(input, loader.GetWords(), save_directory, load_file, pooling);
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
    if ((i = FindArgument(argc, argv, "-layer_sizes")) > 0) { // sizes of the
                                                              // fully-connected
                                                              // layers
      std::stringstream stream(argv[i+1]);
      std::string layer;
      while (getline(stream, layer, '/'))
        layer_sizes.push_back(std::stoi(layer));
    } else
      layer_sizes = {0};
    std::vector<unsigned> layer_sizes_including_input(layer_sizes.size()+1);
    for (unsigned i = 1; i < layer_sizes_including_input.size(); ++i)
      layer_sizes_including_input[i] = layer_sizes[i-1];
    const unsigned epochs = ((i = FindArgument(argc, argv, "-epochs")) > 0)? std::stoi(argv[i+1]) : 40;
    const unsigned mini_batch_size = ((i = FindArgument(argc, argv, "-mini_batch")) > 0)? std::stoi(argv[i+1]) : 0;
    const double lambda = ((i = FindArgument(argc, argv, "-lambda")) > 0)? atof(argv[i+1]) : 0;
    const double learning_rate = ((i = FindArgument(argc, argv, "-eta")) > 0)? atof(argv[i+1]) : 0.1;
    const bool leaky_ReLU_in_convolutional_layers = (((i = FindArgument(argc, argv, "-conv_act")) > 0) && (strcmp(argv[i+1], "1") == 0 || strcmp(argv[i+1], "leaky_relu") == 0))? true : false;
    const unsigned num_of_feature_maps = ((i = FindArgument(argc, argv, "-filters")) > 0)? std::stoi(argv[i+1]) : 8;
    const unsigned num_of_convolutional_and_pooling_layers = ((i = FindArgument(argc, argv, "-conv_layers")) > 0)? std::stoi(argv[i+1]) : 1;
    unsigned activation_function_in_fully_connected_layers = 3; // the default
                                                      // activation function
                                                      // in the fully-connected
                                                      // is tanh
    if ((i = FindArgument(argc, argv, "-fully-connect_act")) > 0) {
      if (strcmp(argv[i+1], "0") == 0 || strcmp(argv[i+1], "relu") == 0)
        activation_function_in_fully_connected_layers = 0;
      else if (strcmp(argv[i+1], "1") == 0 || strcmp(argv[i+1], "leaky_relu") == 0)
        activation_function_in_fully_connected_layers = 1;
      else if (strcmp(argv[i+1], "2") == 0 || strcmp(argv[i+1], "sigmoid") == 0)
        activation_function_in_fully_connected_layers = 2;
    }
    unsigned pooling = 0; // 0 = max pooling
    if ((i = FindArgument(argc, argv, "-pooling")) > 0) {
      if (strcmp(argv[i+1], "1") == 0 || strcmp(argv[i+1], "L2") || strcmp(argv[i+1], "l2") == 0)
        pooling = 1;
      else if (strcmp(argv[i+1], "2") == 0 || strcmp(argv[i+1], "average") == 0)
        pooling = 2;
    }
    unsigned k0 = 3, k1 = 3;
    if ((i = FindArgument(argc, argv, "-kernel_sizes")) > 0) {
      std::stringstream stream(argv[i+1]);
      std::string kernel_size;
      getline(stream, kernel_size, '/');
      k0 = std::stoi(kernel_size);
      getline(stream, kernel_size, '/');
      k1 = std::stoi(kernel_size);
    }
    const bool s_shaped = ((i = FindArgument(argc, argv, "-s_shaped")) > 0 && (strcmp(argv[i+1], "0") || strcmp(argv[i+1], "off") || strcmp(argv[i+1], "false")))? false : true;
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
    Loader loader(data_files, layer_sizes_including_input, s_shaped, {k0, k1});
    const std::vector<std::vector<unsigned>> train_output = loader.LoadTrainOutput();
    const std::vector<std::vector<std::vector<double>>> train_input = loader.ReshapeData(loader.LoadTrainInput());
    const std::vector<std::vector<std::vector<double>>> test_input = loader.ReshapeData(loader.LoadTestInput());
    // Initializes network and starts training...
    ConvolutionalNetwork network(train_input, train_output, test_input, loader.LoadTestOutput(), loader.GetWords(), layer_sizes, num_of_feature_maps, num_of_convolutional_and_pooling_layers, {k0, k1}, pooling, leaky_ReLU_in_convolutional_layers, activation_function_in_fully_connected_layers, epochs, ((mini_batch_size != 0)? mini_batch_size : ((((unsigned) train_output.size()/20+1) < 100)? ((unsigned) train_output.size()/20+1) : 100)), lambda, learning_rate, save_file, load_file, s_shaped);
  }
  return 0;
}
