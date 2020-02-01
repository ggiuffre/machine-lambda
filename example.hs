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
        dataset = take 200 $ shuffled (zip dataMatrices labelsMatrices) randGen

    randGen <- newStdGen
    let net1 = randNet [784, 30, 10] randGen

    randGen <- newStdGen
    let epochs = shuffSgdUpdates net1 dataset 0.25 randGen
        net2 = last $ take 10 epochs

    print $ catAccuracy net2 dataset
    sequence $ [print $ infer m net2 | m <- (take 4 dataMatrices)]
