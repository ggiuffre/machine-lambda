# Machine Lambda

A purely functional implementation of deep neural nets, stochastic gradient descent, and hopefully more.

## Usage

The `DeepNN` Haskell module provides data structures and functions to create, train, and use deep feed-forward neural networks.

To use it, have `DeepNN.hs` in the search path of GHC, then `import DeepNN` inside your Haskell program.

## Dependencies

`DeepNN` only depends on the [`Data.matrix` module](https://hackage.haskell.org/package/matrix-0.3.6.1/docs/Data-Matrix.html). To install it, type `cabal install matrix` in a shell.
