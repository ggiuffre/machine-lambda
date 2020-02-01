module Dataset
( Dataset (..)
, fromCsv
, shuffled
) where



import Data.List (groupBy)
import Data.Char (isSeparator)
import Data.Function (on)
import System.Random (StdGen, random, randomRs)
import Data.Matrix (Matrix, fromLists)



type DataFile = IO String
type Sample t = Matrix t
type Label  t = Matrix t

data Dataset t = DataHeap [Sample t] | DataMap [Sample t] [Label t]



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
