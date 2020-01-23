module DeepNN
( Network
, Layer
, CostFunction
, QuadCost (..)
, CrossEntCost (..)
, infer
, sgdUpdates
, shuffSgdUpdates
, cost
, accuracy
, shuffle
, randNet
) where



import System.Random (StdGen, random, randomRs, mkStdGen)
import Data.Random.Normal (normals')
import Data.List (genericLength)
import Data.Matrix (Matrix, elementwise, fromLists, fromList, toLists, toList, transpose, scaleMatrix)



-- a deep neural network is a sequence of layers
type Network t = [Layer t]

-- each layer of a network has biases and weights
type Layer t = (BiasMatrix t, WeightsMatrix t)

-- the biases and weights of a layer can be represented as matrices
type BiasMatrix t = Matrix t
type WeightsMatrix t = Matrix t

-- a dataset is a list of input-output pairs (input first, label second)
type Dataset t = [(Matrix t, Matrix t)]



-- cost functions can be applied and can be used to compute the output error of a network
class CostFunction f where
    appl :: (Floating t) => f -> (Matrix t -> Matrix t -> t)
    oErr :: (Floating t) => f -> (Matrix t -> Matrix t -> Matrix t)

-- the quadratic cost function is a cost function
data QuadCost = QuadCost
instance CostFunction QuadCost where
    appl costF = f
        where f prediction label = (sum $ foreach (^2) (label - prediction)) / 2
    oErr costF = f'
        where f' zs label = elementwise (*) ((sigm zs) - label) (sigm' zs)
              sigm  = foreach sigmoid
              sigm' = foreach sigmoid'

-- the cross-entropy cost function is a cost function
data CrossEntCost = CrossEntCost
instance CostFunction CrossEntCost where
    appl costF = f
        where f prediction label = - sum (elementwise elementEntropy prediction label)
              elementEntropy a y = (y * log a) + ((1-y) * log (1-a))
    oErr costF = f'
        where f' zs label = (foreach sigmoid zs) - label



-- list of matrices containing the biases of each layer in a given network
biases :: Network t -> [BiasMatrix t]
biases = map fst

-- list of matrices containing the weights of each layer in a given network
weights :: Network t -> [WeightsMatrix t]
weights = map snd

-- sigmoid activation of a given value
sigmoid :: (Floating t) => t -> t
sigmoid x = 1.0 / (1.0 + exp (-x))

-- derivative of the sigmoid activation at a given value
sigmoid' :: (Floating t) => t -> t
sigmoid' x = sigmoid x * (1.0 - sigmoid x)

-- result of applying a function to each element of a matrix
foreach :: (t -> t) -> Matrix t -> Matrix t
foreach func mat = fromLists $ map (map func) $ toLists mat

-- output of a neural network, given some input
infer :: (Floating t) => Matrix t -> Network t -> Matrix t
infer = foldl activation

-- activation and w.ed input of each layer, given an input to the whole network
analyze :: (Floating t) => Matrix t -> Network t -> [(Matrix t, Matrix t)]
analyze input net = zip zs as
    where zs = wdInputs input net            -- weighted inputs of each layer
          as = tail $ scanl activation input net -- activations of each layer

-- weighted input of each layer, given an input to the whole network
wdInputs :: (Floating t) => Matrix t -> Network t -> [Matrix t]
wdInputs input [] = []
wdInputs input net = z:(wdInputs (foreach sigmoid z) (tail net))
    where z = wInput input (head net)

-- activation of one layer, given input, biases, and weights
activation :: (Floating t) => Matrix t -> Layer t -> Matrix t
activation input (biases, weights) = foreach sigmoid (biases + weights * input)

-- weighted input of a layer, given input, biases, and weights
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
          costF = CrossEntCost -- TODO needs to become a parameter
deltas net states label = errWdInput:nextErrs
    where errWdInput = elementwise (*) errActivation (foreach sigmoid' zs)
          errActivation = transpose nextWs * (head nextErrs)
          nextErrs = deltas (tail net) (tail states) label
          (nextBs, nextWs) = head $ tail net
          (zs, as) = head states

-- list of networks whose parameters are updated with (online) SGD throughout an epoch
sgdEpoch :: (Floating t) => Network t -> Dataset t -> t -> [Network t]
sgdEpoch net [] eta = []
sgdEpoch net dataset eta = newNet:nextUpdates
    where nextUpdates = sgdEpoch newNet nextSamples eta
          newNet = zip newBiases newWeights
          newBiases  = zipWith (-) (biases net) correctionsB
          newWeights = zipWith (-) (l2Reg $ weights net) correctionsW
          correctionsB = map (scaleMatrix eta) (gradientBiases  activs errors)
          correctionsW = map (scaleMatrix eta) (gradientWeights activs errors)
          errors = deltas net states label
          (input, label):nextSamples = dataset
          activs = input:(map snd states)
          states = analyze input net

-- L2 regularization of the weights of a network, before updating them with SGD
l2Reg :: (Floating t) => [WeightsMatrix t] -> [WeightsMatrix t]
l2Reg ws = [scaleMatrix (1 - lambda / (fromIntegral $ length w)) w | w <- ws]
    where lambda = 0.002

-- infinite list of networks whose parameters are updated with SGD throughout (infinite) epochs
sgdUpdates :: (Floating t) => Network t -> Dataset t -> t -> [Network t]
sgdUpdates net dataset eta = newNet:nextNets
    where newNet = last $ sgdEpoch net dataset eta
          nextNets = sgdUpdates newNet dataset eta

-- infinite list of networks whose parameters are updated with SGD throughout (infinite) epochs, each time shuffling the dataset
shuffSgdUpdates :: (Floating t) => Network t -> Dataset t -> t -> StdGen -> [Network t]
shuffSgdUpdates net dataset eta gen = newNet:nextNets
    where newNet = last $ sgdEpoch net shuffledDataset eta
          nextNets = shuffSgdUpdates newNet shuffledDataset eta newGen
          shuffledDataset = shuffle dataset gen
          (randInt, newGen) = random gen :: (Int, StdGen)

-- cost of a network on a dataset, for a given cost function
cost :: (CostFunction f, Floating t) => f -> Network t -> Dataset t -> t
cost costF net dataset = (sum costs) / (genericLength costs)
    where costs = [cost input label | (input, label) <- dataset]
          cost input label = (appl costF) (infer input net) label

-- fraction of samples correctly classified by a network
accuracy :: (Floating t, RealFrac t) => Network t -> Dataset t -> t
accuracy net dataset = (sum outcomes) / (genericLength dataset)
    where outcomes = [outcome input label | (input, label) <- dataset]
          outcome x y = boolToNum $ roundedVect (infer x net) == roundedVect y
          roundedVect v = (map round $ toList v)
          boolToNum True = 1.0
          boolToNum _    = 0.0

-- random permutation of the elements in a list, given a rand. number generator
shuffle :: [a] -> StdGen -> [a]
shuffle list gen = if length list < 2 then list else (list!!i : r)
    where i = head $ randomRs (0, length list - 1) newGen :: Int
          r = shuffle (take i list ++ drop (i+1) list) newGen
          (randInt, newGen) = random gen :: (Int, StdGen)

-- network with Double weights sampled uniformly at random from a given range
randUniformNet :: (Double, Double) -> [Int] -> StdGen -> Network Double
randUniformNet range [] _ = []
randUniformNet range (size:[]) _ = []
randUniformNet range sizes gen = randLayer:nextRandLayers
    where randLayer = (randBiases, randWeights)
          nextRandLayers = randUniformNet range (outSize:nextSizes) newGen
          randBiases = fromList outSize 1 $ biasList
          randWeights = fromList outSize inSize $ weightList
          (biasList, weightList) = splitAt outSize randNums
          randNums = take (outSize * (inSize + 1)) $ randomRs range gen
          (_, newGen) = random gen :: (Int, StdGen)
          inSize:outSize:nextSizes = sizes

-- network with Double weights sampled at random from a normal distribution with mean 0 and std. deviation sqrt(n_in) for each layer
randNet :: [Int] -> StdGen -> Network Double
randNet [] _ = []
randNet (size:[]) _ = []
randNet sizes gen = randLayer:nextRandLayers
    where randLayer = (randBiases, randWeights)
          nextRandLayers = randNet (outSize:nextSizes) newGen
          randBiases = fromList outSize 1 $ biasList
          randWeights = fromList outSize inSize $ weightList
          (biasList, weightList) = splitAt outSize randNums
          randNums = take (outSize * (inSize + 1)) $ normals' (0, stdDev) gen
          (_, newGen) = random gen :: (Int, StdGen)
          inSize:outSize:nextSizes = sizes
          stdDev = 1.0 / sqrt (fromIntegral inSize)

-- TODO: mini-batch size, regularization parameter, cost function parameter
