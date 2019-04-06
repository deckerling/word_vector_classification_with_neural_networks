# *word_vector_classification_with_neural_networks*
*word_vector_classification_with_neural_networks* provides neural network implementations for c++ and python to classify word vectors (especially for semantic classifications). The c++-networks can be trained and the trained weights and biases can be loaded later to work with the network on classification problems. (The python implementations lack functions both to save and to load trained biases and weights.)  
The python implementations are based on the *[Keras](https://keras.io/)* library (personnally, I used Google's *[TensorFlow](https://www.tensorflow.org/)* as backend).

## Modes and requirements
All provided networks require at least some input data; test data is optional. If you want to train the network, the training data has to consist of two files: one containing the word vectors and another containing the desired outputs (the same is true for test data).  
If you want to use a pre-trained network to classify word vectors, only a word vector file and a file containing the pre-trained biases and weights are needed.  
The word vector files should contain one word vector per line with all elements of a line being separated by a whitespace. For the c++-networks to work, the first element has to be the word and all others have to be the values of the vector. Of course, all word vectors have to have the same number of dimensions (i.e. the same "vector size"). Suitable word vector files can be created for example with [Stanford's *GloVe* implementation](https://nlp.stanford.edu/projects/glove/). The implementations for python work with files containing only the numeric data – i.e. no "words" should be part of the vector files.  
The files containing the desired outputs should consist of one one-hot vector per line.

### Special notes
*word_vector_classification_with_neural_networks* provides implementations for MLPs and CNNs. Nevertheless, it seems to be not really surprising that I achieved the best results using a MLP for there are not much spatial patterns that can be found the word vectors I used.

## License
The work contained in this package is licensed under the Apache License, Version 2.0 (see the file "[LICENSE](LICENSE)").

