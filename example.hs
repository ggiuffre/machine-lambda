import System.IO
import System.Random (getStdGen, newStdGen)
import DeepNN
import Dataset



-- example use of the DeepNN and Dataset modules
main = do
    csvData <- readFile "xor_train_data.csv"
    csvLabels <- readFile "xor_train_labels.csv"

    randGen <- getStdGen
    let dataMatrices = fromCsv csvData
        labelsMatrices = fromCsv csvLabels
        dataset = zip dataMatrices labelsMatrices

    randGen <- newStdGen
    let net1 = randNet [2, 2, 1] randGen

    randGen <- newStdGen
    let epochs = sgdUpdates net1 dataset 0.25 randGen
        net2 = last $ take 1000 epochs

    print $ binAccuracy net2 dataset
    sequence $ [print $ (y, output x net2) | (x, y) <- dataset]
