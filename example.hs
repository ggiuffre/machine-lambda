import System.IO
import System.Random (StdGen, getStdGen, newStdGen, random)
import qualified Data.Matrix as Mat
import Dataset
import DeepNN



-- example use
main = do
    csvData <- readFile "xor_train_data.csv"
    csvLabels <- readFile "xor_train_labels.csv"
    randGen <- getStdGen
    let dataMatrices = fromCsv csvData
        labelsMatrices = fromCsv csvLabels
        dataset = take 200 $ zip dataMatrices labelsMatrices
        net1 = randNet [2, 2, 1] randGen
    randGen <- newStdGen
    let (randInt, gen) = random randGen :: (Int, StdGen)
        epochs = shuffSgdUpdates net1 dataset 0.25 gen
        net2 = last $ take 1000 epochs
    print $ performance QuadCost net2 dataset
    sequence $ [print $ infer m net2 | m <- dataMatrices]
