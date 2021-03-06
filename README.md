# Machine Lambda

A purely functional Haskell implementation of stochastic gradient descent for deep feed-forward neural networks.

Why functional programming for neural networks? The goal of this library is mathematical readability: the close coupling between functional programming and mathematical expressions allows to manipulate neural networks (or anything that can be expressed mathematically) by declaring expressions, rather than by implementing algorithms.

Approaching a problem by declaring its solution instead of implementing an algorithm to find that solution allows the programmer to be succint and clear. For example, here's the function that computes the output of a neural network for a given input:

```haskell
output :: (Floating t) => Matrix t -> Network t -> Matrix t
output = foldl activation
```

Here, `output` is a function of two arguments: an input `Matrix` and a `Network`. `output` is defined as a [left fold](https://en.wikipedia.org/wiki/Fold_(higher-order_function)) across the layers of the network, with the input matrix as the initial state of the accumulator and the `activation` function as the combining operation.

## Usage

The `example.hs` program shows an example of how to use this library.

The `DeepNN` module exports data structures and functions to create, train, and use deep feed-forward neural networks:

* `Network` is a data type that represents a deep neural network;
* `CostFunction` is the class of cost functions, whose available instances are `QuadCost` and `CrossEntCost`;
* `output` is the output of a neural network, given some input;
* `sgdUpdates` and `sgdUpdates'` are an infinite list of networks whose parameters are updated with SGD throughout (infinite) epochs, respectively with and without re-shuffling the dataset at each epoch;
* `cost` is the cost of a neural network on a dataset, w.r.t. a given cost function;
* `binAccuracy` and `catAccuracy` are the accuracies (resp. binary and categorical) of a neural network on a dataset;
* `randNet` is a network with random `Double` weights sampled from a normal distribution with given mean and standard deviation.

The `Dataset` module currently exports the following functions:

* `fromCsv` is a dataset of 1D `Double` samples, taken from a given CSV file;
* `shuffled` is a random permutation of the elements in a list, given a random number generator;
* `foreach` is the result of applying a function to each element of a matrix;
* `standardized` is a list of training samples that have been standardized.

To use a module (such as `DeepNN` for example), have `DeepNN.hs` in the search path of GHC, then `import DeepNN` inside your Haskell program. See the example program.

## Dependencies

`DeepNN` depends on several built-in Haskell modules, but also on the [`Data.matrix` module](https://hackage.haskell.org/package/matrix-0.3.6.1/docs/Data-Matrix.html): to install it, type `cabal install matrix` in a shell.
