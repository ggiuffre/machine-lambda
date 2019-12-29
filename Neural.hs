module Neural
( Network
, Layer
, infer
, analyze
, getErrors
, sgdEpoch
, sgdUpdates
, shuffle
) where

import System.Random (randomRIO)
import Data.List (genericLength)
import Data.Matrix (Matrix, elementwise, fromLists, toLists, transpose, scaleMatrix)

type BiasMatrix t = Matrix t
type WeightsMatrix t = Matrix t
type Layer t = (BiasMatrix t, WeightsMatrix t)
type Network t = [Layer t]

biases net = map fst net
weights net = map snd net

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
infer input net = foldl activation input net

-- activation and w.ed input of each layer, given an input to the whole network
analyze :: (Floating t) => Matrix t -> Network t -> [(Matrix t, Matrix t)]
analyze input net = zip zs as
    where zs = wdInputs input net
          as = tail $ scanl activation input net

-- weighted input of each layer, given an input to the whole network
wdInputs :: (Floating t) => Matrix t -> Network t -> [Matrix t]
wdInputs input [] = []
wdInputs input net = z:(wdInputs (foreach sigmoid z) (tail net))
    where z = wInput input (head net)

-- activation of a layer given input, weights, and biases
activation :: (Floating t) => Matrix t -> Layer t -> Matrix t
activation input (biases, weights) = foreach sigmoid (biases + weights * input)

-- weighted input of a layer given input, weights, and biases
wInput :: (Floating t) => Matrix t -> Layer t -> Matrix t
wInput input (biases, weights) = biases + weights * input

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
getErrors :: (Floating t) => Network t -> [(Matrix t, Matrix t)] -> Matrix t -> [Matrix t]
getErrors (l:[]) states label = [errWdInput]
    where errWdInput = elementwise (*) errActivation (foreach sigmoid' zs)
          errActivation = (as - label)
          (zs, as) = head states
getErrors net states label = errWdInput:nextErrs
    where errWdInput = elementwise (*) errActivation (foreach sigmoid' zs)
          errActivation = transpose nextWs * (head nextErrs)
          nextErrs = getErrors (tail net) (tail states) label
          (nextBs, nextWs) = head $ tail net
          (zs, as) = head states

sgdEpoch :: (Floating t) => Network t -> [(Matrix t, Matrix t)] -> t -> [Network t]
sgdEpoch net [] eta = []
sgdEpoch net dataset eta = newNet:nextUpdates
    where nextUpdates = sgdEpoch newNet (tail dataset) eta
          newNet = zip newBiases newWeights
          newBiases  = map matsDiff (zip (biases  net) correctionsB)
          newWeights = map matsDiff (zip (weights net) correctionsW)
          matsDiff (m1, m2) = m1 - m2
          correctionsB = map (scaleMatrix eta) (gradientBiases  activs errors)
          correctionsW = map (scaleMatrix eta) (gradientWeights activs errors)
          errors = getErrors net states label
          input  = fst $ head dataset
          label  = snd $ head dataset
          activs = input:(map snd states)
          states = analyze input net

-- infinite list of networks whose parameters are changed with SGD
sgdUpdates :: (Floating t) => Network t -> [(Matrix t, Matrix t)] -> t -> [Network t]
sgdUpdates net dataset eta = newNet:nextNets
    where newNet = last $ sgdEpoch net dataset eta
          nextNets = sgdUpdates newNet dataset eta

-- average performance of a network on a dataset
performance :: (Floating t) => Network t -> [(Matrix t, Matrix t)] -> t
performance net dataset = sum costs / genericLength costs
    where costs = [cost input label | (input, label) <- dataset]
          cost input label = quadCost (infer input net) label

shuffle x = if length x < 2 then return x else do
    i <- System.Random.randomRIO (0, length(x)-1)
    r <- shuffle (take i x ++ drop (i+1) x)
    return (x!!i : r)
