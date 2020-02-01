# Machine Lambda

A purely functional implementation of a deep feed-forward neural network, trainable with stochastic gradient descent.

## Usage

The `example.hs` program shows an example of how to use this library.

The `DeepNN` module exports data structures and functions to create, train, and use deep feed-forward neural networks:

* `Network` and `Layer` are types that represent respectively a deep neural network and a single layer;
* `CostFunction` is the class of cost functions, whose available instances are `QuadCost` and `CrossEntCost`;
* `infer` is the output of a neural network, given some input;
* `sgdUpdates` and `sgdUpdates'` are an infinite list of networks whose parameters are updated with SGD throughout (infinite) epochs, respectively with and without re-shuffling the dataset at each epoch;
* `cost` is the cost of a neural network on a dataset, w.r.t. a given cost function;
* `binAccuracy` and `catAccuracy` are the accuracies (resp. binary and categorical) of a neural network on a dataset;
* `randNet` is a network with random `Double` weights sampled from a normal distribution with given mean and standard deviation.

The `Dataset` module currently exports two functions:

* `fromCsv` is a dataset of 1D `Double` samples, taken from a given CSV file;
* `shuffled` is a random permutation of the elements in a list, given a random number generator.

To use a module (such as `DeepNN` for example), have `DeepNN.hs` in the search path of GHC, then `import DeepNN` inside your Haskell program.

## Dependencies

`DeepNN` only depends on the [`Data.matrix` module](https://hackage.haskell.org/package/matrix-0.3.6.1/docs/Data-Matrix.html). To install it, type `cabal install matrix` in a shell.
