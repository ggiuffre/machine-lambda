module Dataset
( Dataset (..)
, fromCsv
) where



import Data.Char (isSeparator)
import Data.Function (on)
import Data.List (groupBy)
import Data.Matrix (Matrix, fromLists)



type DataFile = IO String
type Sample t = Matrix t
type Label  t = Matrix t

data Dataset t = DataHeap [Sample t] | DataMap [Sample t] [Label t]



fromCsv :: String -> [Matrix Float]
fromCsv samples = [matrix $ values line | line <- lines samples]
    where values l = filter isValue $ groupBy ((==) `on` not . isDelimiter) $ l
          isValue = not . any isDelimiter
          isDelimiter c = c == ',' || isSeparator c
          matrix l = fromLists $ map parseRow l
          parseRow r = [read r :: Float]
