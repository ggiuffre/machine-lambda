import System.IO
import System.Random (StdGen, getStdGen, newStdGen, random)
import qualified Data.Matrix as Mat
import Dataset
import DeepNN



main = do
    csvData <- readFile "xor_train_data.csv"
    csvLabels <- readFile "xor_train_labels.csv"
    randGen <- getStdGen
    let dataMatrices = fromCsv csvData
        net1 = randNet (-0.8, 0.8) [2, 2, 1] randGen
        labelsMatrices = fromCsv csvLabels
        dataset = zip dataMatrices labelsMatrices
    randGen <- newStdGen
    let (randInt, gen) = random randGen :: (Int, StdGen)
        epochs = shuffSgdUpdates net1 dataset 0.8 gen
        net2 = last $ take 5000 epochs
    sequence $ map (\x -> print $ infer x net2) dataMatrices
