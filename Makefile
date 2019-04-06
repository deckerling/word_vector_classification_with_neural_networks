CFLAGS := -g -Wall
BUILDDIR := build

all: builddir mlp4wordvec cnn4wordvec

builddir:
	mkdir -p $(BUILDDIR)
mlp4wordvec: $(OBJS)
	g++ src/cpp/mlp_main.cc src/cpp/data_loader.h src/cpp/data_loader.cc src/cpp/mlp.h src/cpp/multilayer_perceptron.cc src/cpp/neural_network.h src/cpp/neural_network.cc -o $(BUILDDIR)/mlp4wordvec $(CFLAGS)
cnn4wordvec: $(OBJS)
	g++ src/cpp/cnn_main.cc src/cpp/data_loader.h src/cpp/data_loader.cc src/cpp/cnn.h src/cpp/convolutional_neural_network.cc src/cpp/neural_network.h src/cpp/neural_network.cc -o $(BUILDDIR)/cnn4wordvec $(CFLAGS)

clean:
	rm -rf mlp4wordvec cnn4wordvec build
