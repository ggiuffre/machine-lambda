module DeepNN
( Network
, Layer
, infer
, sgdUpdates
, performance
, shuffle
) where



import System.Random (randomRs, mkStdGen)
import Data.List (genericLength)
import Data.Matrix (Matrix, elementwise, fromLists, toLists, transpose, scaleMatrix)



-- a deep neural network is a sequence of layers
type Network t = [Layer t]

-- each layer of a network has biases and weights
type Layer t = (BiasMatrix t, WeightsMatrix t)

-- the biases and weights of a layer can be represented as matrices
type BiasMatrix t = Matrix t
type WeightsMatrix t = Matrix t

-- a dataset is a list of input-output pairs (input first, label second)
type Dataset t = [(Matrix t, Matrix t)]



-- a class for cost functions
class CostFunction f where
    appl :: (Floating t) => f -> (Matrix t -> Matrix t -> t)
    oErr :: (Floating t) => f -> (Matrix t -> Matrix t -> Matrix t)

data QuadCost = QuadCost
data CrossEntCost = CrossEntCost

instance CostFunction QuadCost where
    appl costF = f
        where f prediction label = (sum (label - prediction) ^ 2) / 2
    oErr costF = f'
        where f' zs label = elementwise (*) ((sigm zs) - label) (sigm' zs)
              sigm  = foreach sigmoid
              sigm' = foreach sigmoid'

instance CostFunction CrossEntCost where
    appl costF = f
        where f prediction label = - sum (elementwise elementEntropy prediction label)
              elementEntropy a y = (y * log a) + ((1-y) * log (1-a))
    oErr costF = f'
        where f' zs label = (foreach sigmoid zs) - label



-- list of matrices containing biases, layer by layer
biases :: Network t -> [BiasMatrix t]
biases net = map fst net

-- list of matrices containing weights, layer by layer
weights :: Network t -> [BiasMatrix t]
weights net = map snd net

-- sigmoid of a given number
sigmoid :: (Floating t) => t -> t
sigmoid x = 1.0 / (1.0 + exp (-x))

-- derivative of the sigmoid at a given value
sigmoid' :: (Floating t) => t -> t
sigmoid' x = sigmoid x * (1.0 - sigmoid x)

-- result of applying a function to each element of a matrix
foreach :: (t -> t) -> Matrix t -> Matrix t
foreach func mat = fromLists $ map (map func) $ toLists mat

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

-- partial derivatives of a cost function w.r.t. the weighted inputs of layers in a network
deltas :: (Floating t) => Network t -> [(Matrix t, Matrix t)] -> Matrix t -> [Matrix t]
deltas (l:[]) states label = [errWdInput]
    where errWdInput = (oErr costF) zs label
          (zs, as) = head states
          costF = CrossEntCost -- TODO needs to become a parameter of "deltas"
deltas net states label = errWdInput:nextErrs
    where errWdInput = elementwise (*) errActivation (foreach sigmoid' zs)
          errActivation = transpose nextWs * (head nextErrs)
          nextErrs = deltas (tail net) (tail states) label
          (nextBs, nextWs) = head $ tail net
          (zs, as) = head states

-- list of networks whose parameters are updated with SGD throughout an epoch
sgdEpoch :: (Floating t) => Network t -> Dataset t -> t -> [Network t]
sgdEpoch net [] eta = []
sgdEpoch net dataset eta = newNet:nextUpdates
    where nextUpdates = sgdEpoch newNet (tail dataset) eta
          newNet = zip newBiases newWeights
          newBiases  = map matsDiff (zip (biases  net) correctionsB)
          newWeights = map matsDiff (zip (weights net) correctionsW)
          matsDiff (m1, m2) = m1 - m2
          correctionsB = map (scaleMatrix eta) (gradientBiases  activs errors)
          correctionsW = map (scaleMatrix eta) (gradientWeights activs errors)
          errors = deltas net states label
          input  = fst $ head dataset
          label  = snd $ head dataset
          activs = input:(map snd states)
          states = analyze input net

-- infinite list of networks whose parameters are changed with SGD throughout (infinite) epochs
sgdUpdates :: (Floating t) => Network t -> Dataset t -> t -> [Network t]
sgdUpdates net dataset eta = newNet:nextNets
    where newNet = last $ sgdEpoch net dataset eta
          nextNets = sgdUpdates newNet dataset eta

-- quadratic cost of a network on a dataset
performance :: (CostFunction f, Floating t) => f -> Network t -> Dataset t -> t
performance costF net dataset = sum costs / genericLength costs
    where costs = [cost input label | (input, label) <- dataset]
          cost input label = (appl costF) (infer input net) label

-- TODO
shuffle list seed = if length list < 2 then list else (list!!i : r)
    where i = head $ randomRs (0, length list - 1) (mkStdGen seed) :: Int
          r = shuffle (take i list ++ drop (i+1) list) (seed + 1)
