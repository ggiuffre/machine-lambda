# Machine Lambda

A purely functional implementation of deep neural nets, stochastic gradient descent, and hopefully more.

## Usage

The file `example.hs` shows how to use this software.

The `DeepNN` module exports data structures and functions to create, train, and use deep feed-forward neural networks:

* `Network` and `Layer` are types that represent respectively a deep neural network and a single layer;
* `CostFunction` is the class of cost functions, whose available instances are `QuadCost` and `CrossEntCost`;
* `infer` is the output of a neural network, given some input;
* `sgdUpdates` and `shuffSgdUpdates` are an infinite list of networks whose parameters are updated with SGD throughout (infinite) epochs, respectively with and without shuffling of the dataset at each epoch;
* `performance` is the cost of a neural network on a dataset, for a given cost function;
* `shuffle` is a utility function to shuffle a list of things;
* `randNet` is a network with `Float` weights sampled uniformly at random from a given range.

The `Dataset` module currently exports just one function:

* `fromCsv` is a dataset of 1D `Float` samples, taken from a given CSV file.

To use a module (`DeepNN` for example), have `DeepNN.hs` in the search path of GHC, then `import DeepNN` inside your Haskell program.

## Dependencies

`DeepNN` only depends on the [`Data.matrix` module](https://hackage.haskell.org/package/matrix-0.3.6.1/docs/Data-Matrix.html). To install it, type `cabal install matrix` in a shell.
