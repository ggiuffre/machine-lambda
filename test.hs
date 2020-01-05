import System.IO
import Dataset
import DeepNN

main = do
    dataFile <- openFile "mnist_train_data.csv" ReadMode
    labelFile <- openFile "mnist_train_labels.csv" ReadMode
    csvData <- hGetContents dataFile
    csvLabels <- hGetContents labelFile
    let dataMatrices = fromCsv csvData
    let labelsMatrices = fromCsv csvLabels
    let dataset = DataMap dataMatrices labelsMatrices
    hClose dataFile
    hClose labelFile
