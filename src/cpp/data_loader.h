// data_loader.h

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

#ifndef WORD_VECTOR_CLASSIFICATION_WITH_NEURAL_NETWORKS_SRC_CPP_DATA_LOADER_H_
#define WORD_VECTOR_CLASSIFICATION_WITH_NEURAL_NETWORKS_SRC_CPP_DATA_LOADER_H_

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

inline bool FileIsValid(const std::string& file_path) {
  // Checks if a file can be read.
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cout << file_path << " couldn't be opened!" << '\n';
    return false;
  } else if (file.bad()) {
    std::cout << file_path << " is damaged!" << '\n';
    return false;
  }
  return true;
}

class Loader {
 public:
  // Constructors for MLPs:
  //   Constructor for a network to be trained:
  Loader(const std::vector<std::string>& files, const std::vector<unsigned>& layer_sizes);
  //   Constructor for a pre-trained network that should do its duty:
  Loader(const std::string& input_file);
  // Constructors for CNNs:
  //   Constructor for a network to be trained:
  Loader(const std::vector<std::string>& files, const std::vector<unsigned>& layer_sizes, const bool s_shaped, const std::vector<unsigned>& kernel_sizes);
  //   Constructor for a pre-trained network that should do its duty:
  Loader(const std::string& input_file, const std::string& load_file);

  std::vector<std::vector<double>> LoadTrainInput();
  std::vector<std::vector<unsigned>> LoadTrainOutput();
  std::vector<std::vector<double>> LoadTestInput();
  std::vector<unsigned> LoadTestOutput();
  std::vector<std::vector<std::string>> GetWords() {
    return words;
  }
  std::vector<unsigned> GetFinalLayerSizes() {
    return final_layer_sizes;
  }
  std::vector<std::vector<std::vector<double>>> ReshapeData(const std::vector<std::vector<double>>& vectors);

 private:
  const std::string train_input_file_, train_output_file_, test_input_file_, test_output_file_;
  const std::vector<unsigned> data_size_;
  const bool s_shaped_;
  const std::vector<unsigned> kernel_sizes_;

  std::vector<unsigned> final_layer_sizes;
  std::vector<std::vector<std::string>> words;

  std::vector<unsigned> CountWordVectorsAndTheirElements();
  bool GetInputShape(const std::string& load_file);
  std::vector<unsigned> GetKernelSizes(const std::string& load_file);

  template <typename T>
  void SplitLine(T& values, std::string& line, const char delimiter, const unsigned parseIndex);

  unsigned DivisableOnlyByPrimes(const std::vector<unsigned>& primes, const unsigned vector_size);
};

#endif // WORD_VECTOR_CLASSIFICATION_WITH_NEURAL_NETWORKS_SRC_CPP_DATA_LOADER_H_
