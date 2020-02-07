module Dataset
( Dataset (..)
, fromCsv
, shuffled
, foreach
, standardized
) where



import Data.List (groupBy)
import Data.Char (isSeparator)
import Data.Function (on)
import System.Random (StdGen, random, randomRs)
import Data.Matrix (Matrix, fromLists, toLists, (<|>))



-- in a dataset, each sample and label is represented as a matrix
type Sample t = Matrix t
type Label  t = Matrix t

-- a dataset is a list of relations between samples and labels
type Dataset t = [(Sample t, Label t)]



-- dataset of 1D Double samples, taken from a given CSV file
fromCsv :: String -> [Matrix Double]
fromCsv samples = [matrix $ values line | line <- lines samples]
    where values l = filter isValue $ groupBy ((==) `on` not . isDelimiter) $ l
          isValue = not . any isDelimiter
          isDelimiter c = c == ',' || isSeparator c
          matrix l = fromLists $ map parseRow l
          parseRow r = [read r :: Double]

-- random permutation of the elements in a list, given a rand. number generator
shuffled :: [a] -> StdGen -> [a]
shuffled list gen = if length list < 2 then list else (list!!i : r)
    where i = head $ randomRs (0, length list - 1) newGen :: Int
          r = shuffled (take i list ++ drop (i+1) list) newGen
          (randInt, newGen) = random gen :: (Int, StdGen)

-- result of applying a function to each element of a matrix
foreach :: (t -> t) -> Matrix t -> Matrix t
foreach func mat = fromLists $ map (map func) $ toLists mat

-- standardized version of a list of matrices
standardized :: Floating a => [Sample a] -> [Sample a]
standardized mats = map (foreach center) mats
    where center x = (x - mu) / sigma
          elements = foldl1 (<|>) mats
          mu = avg elements
          sigma = stdDev elements

-- average of the elements of a matrix
avg :: Fractional a => Matrix a -> a
avg mat = sum mat / (fromIntegral $ length mat)

-- standard deviation of the elements of a matrix
stdDev :: Floating a => Matrix a -> a
stdDev = sqrt . variance

-- variance of the elements of a matrix
variance :: Floating a => Matrix a -> a
variance mat = sum squaredDeviations / (n - 1.0)
    where squaredDeviations = foreach (\m -> (m - mu) ^ 2) mat
          n = fromIntegral $ length mat
          mu = avg mat
