import System.IO
import System.Random (StdGen, getStdGen, newStdGen, random)
import qualified Data.Matrix as Mat
import Dataset
import DeepNN



-- example use
main = do
    csvData <- readFile "mnist_train_data.csv"
    csvLabels <- readFile "mnist_train_labels.csv"
    randGen <- getStdGen
    let dataMatrices = fromCsv csvData
        labelsMatrices = fromCsv csvLabels
        dataset = take 200 $ zip dataMatrices labelsMatrices
        net1 = randNet [784, 30, 10] randGen
    randGen <- newStdGen
    let (randInt, gen) = random randGen :: (Int, StdGen)
        epochs = shuffSgdUpdates net1 dataset 0.25 gen
        net2 = last $ take 10 epochs
    print $ accuracy net2 dataset
    -- sequence $ [print $ infer m net2 | m <- dataMatrices]
