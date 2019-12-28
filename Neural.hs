module Neural
( Network (..)
, infer
, analyze
, getErrors
, gradientWeights
, gradientBiases
) where

import Data.Matrix (Matrix, elementwise, fromLists, toLists, transpose)

data Network t = Network { biases :: [Matrix t]
                         , weights :: [Matrix t]
                         } deriving (Show)

-- sigmoid activation function
sigmoid :: (Floating t) => t -> t
sigmoid x = 1.0 / (1.0 + exp (-x))

-- derivative of the sigmoid activation function
sigmoid' :: (Floating t) => t -> t
sigmoid' x = sigmoid x * (1.0 - sigmoid x)

-- apply a function to each element of a matrix
foreach :: (t -> t) -> Matrix t -> Matrix t
foreach func mat = fromLists $ map (map func) $ toLists mat

-- quadratic cost function
quadCost :: (Floating t) => Matrix t -> Matrix t -> t
quadCost out label = sum (foreach (^2) (label - out)) / 2

-- derivative of the quadratic cost function w.r.t.
-- the activation of one output neuron
quadCost' :: (Floating t) => Matrix t -> Matrix t -> Matrix t
quadCost' out label = out - label

-- output of a neural network given some input
infer :: (Floating t) => Matrix t -> Network t -> Matrix t
infer input net = foldl activation input layers
    where layers = zip (weights net) (biases net)

-- activation and w.ed input of each layer, given an input to the whole network
analyze :: (Floating t) => Matrix t -> Network t -> [(Matrix t, Matrix t)]
analyze input net = zip zs as
    where layers = zip (weights net) (biases net)
          zs = wdInputs input layers
          as = tail $ scanl activation input layers

-- weighted input of each layer, given an input to the whole network
wdInputs :: (Floating t) => Matrix t -> [(Matrix t, Matrix t)] -> [Matrix t]
wdInputs input [] = []
wdInputs input layers = z:(wdInputs (foreach sigmoid z) (tail layers))
    where z = wInput input (head layers)

-- activation of a layer given input, weights, and biases
activation :: (Floating t) => Matrix t -> (Matrix t, Matrix t) -> Matrix t
activation input (weights, biases) = foreach sigmoid (biases + weights * input)

-- weighted input of a layer given input, weights, and biases
wInput :: (Floating t) => Matrix t -> (Matrix t, Matrix t) -> Matrix t
wInput input (weights, biases) = biases + weights * input

-- gradient of a cost function w.r.t. the weights of a network
gradientWeights :: (Floating t) => [Matrix t] -> [Matrix t] -> [Matrix t]
gradientWeights inputs [] = []
gradientWeights inputs errors = layerGradient:nextGradients
    where layerGradient = layerError * transpose layerInput
          nextGradients = gradientWeights nextInputs nextErrors
          layerInput:nextInputs = inputs
          layerError:nextErrors = errors

-- gradient of a cost function w.r.t. the biases of a network
gradientBiases :: (Floating t) => [Matrix t] -> [Matrix t] -> [Matrix t]
gradientBiases inputs errors = errors

-- error of the weighted inputs of a network w.r.t. a cost function
getErrors :: (Floating t) => [(Matrix t, Matrix t)] -> [(Matrix t, Matrix t)] -> Matrix t -> [Matrix t]
getErrors (l:[]) states label = [errWdInput]
    where errWdInput = elementwise (*) errActivation (foreach sigmoid' zs)
          errActivation = (as - label)
          (zs, as) = head states
getErrors layers states label = errWdInput:nextErrs
    where errWdInput = elementwise (*) errActivation (foreach sigmoid' zs)
          errActivation = transpose nextWs * (head nextErrs)
          nextErrs = getErrors (tail layers) (tail states) label
          (nextWs, nextBs) = head $ tail layers
          (zs, as) = head states

sgd :: (Floating t) => Network t -> [(Matrix t, Matrix t)] -> Network t
sgd net dataset = dropWhile badPerformance nets
    where badPerformance n = performance n dataset < 1 -- ??
          nets = [Network bs ws | (bs, ws) <- paramUpdates]
          paramUpdates = idontknow -- ??

performance :: (Floating t) => Network t -> [(Matrix t, Matrix t)] -> t
performance net dataset = sum costs / length costs
    where costs = [quadCost out label | (input, label) <- dataset]
          out = infer input net