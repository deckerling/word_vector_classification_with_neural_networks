// data_loader.cc

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
#include <cmath>
#include <sstream>

#include "data_loader.h"

Loader::Loader(const std::vector<std::string>& files, const std::vector<unsigned>& layer_sizes)
    : train_input_file_(files[0]),
      train_output_file_(files[1]),
      test_input_file_(files[2]),
      test_output_file_(files[3]),
      data_size_(CountWordVectorsAndTheirElements()),
      s_shaped_(false),
      kernel_sizes_({}) {
  std::cout << "Loading data..." << std::endl;
  final_layer_sizes = layer_sizes;
  words = std::vector<std::vector<std::string>>(2);
  words[0].reserve(data_size_[0]);
  words[1].reserve(data_size_[1]);
  if (data_size_[2] != final_layer_sizes[0]) { // if the number of neurons of
                                               // the input layer desired by
                                               // the user is not equal to the
                                               // size of the given data
    final_layer_sizes[0] = data_size_[2];
    std::cout << '\t' << "WARNING: The number of elements of the word vectors is not equal to your selected number of input neurons!" << '\n' << '\t' << "Thus, the number of input neurons will be replaced with the number of elements of the word vectors (nevertheless, errors may occur)." << '\n';
  }
}

Loader::Loader(const std::string& input_file)
    : train_input_file_(input_file),
      train_output_file_(""),
      test_input_file_(""),
      test_output_file_(""),
      data_size_(CountWordVectorsAndTheirElements()),
      s_shaped_(false),
      kernel_sizes_({}) {
  std::cout << "Loading data..." << std::endl;
  words = std::vector<std::vector<std::string>>(1);
  words[0].reserve(data_size_[0]);
}

Loader::Loader(const std::vector<std::string>& files, const std::vector<unsigned>& layer_sizes, const bool s_shaped, const std::vector<unsigned>& kernel_sizes)
    : train_input_file_(files[0]),
      train_output_file_(files[1]),
      test_input_file_(files[2]),
      test_output_file_(files[3]),
      data_size_(CountWordVectorsAndTheirElements()),
      s_shaped_(s_shaped),
      kernel_sizes_(kernel_sizes) {
  std::cout << "Loading data..." << std::endl;
  final_layer_sizes = layer_sizes;
  words = std::vector<std::vector<std::string>>(2);
  words[0].reserve(data_size_[0]);
  words[1].reserve(data_size_[1]);
  if (data_size_[2] != final_layer_sizes[0]) { // if the number of neurons of
                                               // the input layer desired by
                                               // the user is not equal to the
                                               // size of the given data
    final_layer_sizes[0] = data_size_[2];
  }
}

Loader::Loader(const std::string& input_file, const std::string& load_file)
    : train_input_file_(input_file),
      train_output_file_(""),
      test_input_file_(""),
      test_output_file_(""),
      data_size_(CountWordVectorsAndTheirElements()),
      s_shaped_(GetInputShape(load_file)),
      kernel_sizes_(GetKernelSizes(load_file)) {
  std::cout << "Loading data..." << std::endl;
  words = std::vector<std::vector<std::string>>(1);
  words[0].reserve(data_size_[0]);
}

std::vector<unsigned> Loader::CountWordVectorsAndTheirElements() {
  // Counts the word vectors and their elements assuming that each line of the
  // word vector file contains exactly one word vector with all values
  // separated with whitespaces and the first "value" being the "word".
  // Furthermore, all vectors should have the same size. This applies both the
  // training and the test vector files (if test data is provided at all).
  std::ifstream train_input_file(train_input_file_);
  std::string line;
  std::vector<unsigned> data_size(3, 0);
  while (std::getline(train_input_file, line)) {
    if (data_size[0] == 0)
      data_size[2] = std::count(line.begin(), line.end(), ' ');
    data_size[0]++;
  }
  if (!test_output_file_.empty()) {
    std::ifstream test_output_file(test_output_file_);
    while (std::getline(test_output_file, line))
      data_size[1]++;
  }
  return data_size;
}

bool Loader::GetInputShape(const std::string& load_file) {
  // Checks whether the weights in the "load_file" apply to "s_shaped" data or
  // not.
  std::ifstream load_file_stream(load_file);
  std::string line;
  std::getline(load_file_stream, line);
  std::getline(load_file_stream, line);
  return std::stoi(line);
}

std::vector<unsigned> Loader::GetKernelSizes(const std::string& load_file) {
  // Returns the "kernel_sizes" found in the "load_file".
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

std::vector<std::vector<double>> Loader::LoadTrainInput() {
  std::ifstream train_input_file(train_input_file_);
  std::string line;
  std::vector<double> values(data_size_[2]);
  std::vector<std::vector<double>> train_input(data_size_[0], std::vector<double>(data_size_[2]));
  for (auto& input : train_input) {
    if (std::getline(train_input_file, line))
      SplitLine(values, line, ' ', 0);
    input = values;
  }
  return train_input;
}

std::vector<std::vector<unsigned>> Loader::LoadTrainOutput() {
  // The output data should only contain one-hot vectors.
  std::ifstream train_output_file(train_output_file_);
  std::string line;
  std::getline(train_output_file, line);
  if (final_layer_sizes[final_layer_sizes.size()-1] == 0)
    final_layer_sizes[final_layer_sizes.size()-1] = (std::count(line.begin(), line.end(), ' ')+1);
  std::vector<unsigned> values(final_layer_sizes[final_layer_sizes.size()-1]);
  std::vector<std::vector<unsigned>> train_output(data_size_[0], std::vector<unsigned>(final_layer_sizes[final_layer_sizes.size()-1]));
  for (unsigned i = 0; i < data_size_[0]; ++i) {
    if (i != 0)
      std::getline(train_output_file, line);
    SplitLine(values, line, ' ', 2);
    train_output[i] = values;
  };
  return train_output;
}

std::vector<std::vector<double>> Loader::LoadTestInput() {
  if (data_size_[1] != 0 && !test_input_file_.empty()) {
    std::ifstream test_input_file(test_input_file_);
    std::string line;
    std::vector<double> values(data_size_[2]);
    std::vector<std::vector<double>> test_input(data_size_[1], std::vector<double>(data_size_[2]));
    for (auto& input : test_input) {
      if (std::getline(test_input_file, line))
        SplitLine(values, line, ' ', 1);
      input = values;
    }
    return test_input;
  }
  return {{}};
}

std::vector<unsigned> Loader::LoadTestOutput() {
  // The output data should only contain one-hot vectors.
  if (data_size_[1] != 0) {
    std::vector<unsigned> test_output;
    test_output.reserve(data_size_[1]);
    std::ifstream test_output_file(test_output_file_);
    std::string line, value;
    unsigned index;
    while (std::getline(test_output_file, line)) {
      std::stringstream stream(line);
      index = 0;
      while (getline(stream, value, ' '))
        if (value == "1") {
          test_output.push_back(index); // only the index of the one-hot vector
                                        // will be used
          break;
        } else
          index++;
    }
    return test_output;
  }
  return {};
}

template <typename T>
void Loader::SplitLine(T& values, std::string& line, const char delimiter, const unsigned parseIndex) {
  // Splits a line of a data file and loads the values into the std::vector<T>
  // "values".
  std::stringstream stream(line);
  std::string value;
  if (parseIndex != 2) { // parseIndex == 0 => input training data is given;
                         // parseIndex == 1 => input test data is given; else
                         // => output data is given (the output data contains
                         // no "words")
    getline(stream, value, delimiter);
    words[parseIndex].push_back(value); // "words[0]" will contain the words of
                                        // the training data, "words[1]" those
                                        // of the test data
    for (auto& element : values) {
      getline(stream, value, delimiter);
      element = atof(value.c_str());
    }
    return;
  }
  for (auto& element : values) {
    getline(stream, value, delimiter);
    element = std::stoi(value);
  }
}

std::vector<std::vector<std::vector<double>>> Loader::ReshapeData(const std::vector<std::vector<double>>& vectors) {
// Checks if the "vectors[0].size()" is a prime number (assuming that
// vectors[0].size() < 1000 && vectors[0].size() > 2); then sets the size of
// the two dimensions of the input layer in a suitable way, reshapes the given
// vectors and returns them.
  const unsigned vector_size = vectors[0].size();
  const std::vector<unsigned> primes = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997};
  unsigned x = 0;
  for (auto& prime : primes) {
    if (prime > vector_size)
      break;
    if (prime == vector_size) {
      x++; // will add a 0 to the end of the vector in order to make it
           // possible to transform the vector into a "rectangle"
      break;
    }
  }
  unsigned first_dimension, l;
  if ((first_dimension = DivisableOnlyByPrimes(primes, (vector_size+x))) == 0)
    first_dimension = std::sqrt(vector_size+x);
  first_dimension = first_dimension+(kernel_sizes_[0]-1); // "+(kernel_sizes[0]-1)" makes sure that zero padding is used
  const unsigned second_dimension = (vector_size+x)/first_dimension+(kernel_sizes_[1]-1); // "+(kernel_sizes[1]-1)" makes sure that zero padding is used
  std::vector<std::vector<std::vector<double>>> reshaped_vectors = std::vector<std::vector<std::vector<double>>>(vectors.size(), std::vector<std::vector<double>>(first_dimension, std::vector<double>(second_dimension)));
  for (unsigned i = 0; i < vectors.size(); ++i) {
    l = 0;
    for (unsigned j = (kernel_sizes_[0]-1)/2; j < (first_dimension-(kernel_sizes_[0]-1)/2); ++j) {
      if (s_shaped_) {
        for (unsigned k = (((j%2) == 0)? (kernel_sizes_[1]-1)/2 : ((second_dimension-(kernel_sizes_[1]-1)/2)-1)); (((j%2) == 0)? (k < second_dimension-(kernel_sizes_[1]-1)/2) : k > ((kernel_sizes_[1]-1)/2-1)); (((j%2) == 0)? ++k : --k)) {
          if (l < vector_size) {
            reshaped_vectors[i][j][k] = vectors[i][l];
            l++;
          }
        }
      } else {
        for (unsigned j = (kernel_sizes_[0]-1)/2; j < (first_dimension-(kernel_sizes_[0]-1)/2); ++j) {
          for (unsigned k = (kernel_sizes_[1]-1)/2; k < (second_dimension-(kernel_sizes_[1]-1)/2); ++k) {
            if (l < vector_size) {
              reshaped_vectors[i][j][k] = vectors[i][l];
              l++;
            }
          }
        }
      }
    }
  }
  return reshaped_vectors;
}

unsigned Loader::DivisableOnlyByPrimes(const std::vector<unsigned>& primes, const unsigned vector_size) {
  for (auto& prime0 : primes) {
    if (prime0 > (vector_size/2))
      return 0;
    if ((vector_size%prime0) == 0) {
      for (auto& prime1 : primes) {
        if ((vector_size/prime0) == prime1)
          return prime0;
      }
      return 0;
    }
  }
  return 0;
}
