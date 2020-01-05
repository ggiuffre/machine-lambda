import System.IO
import Dataset
import DeepNN

main = do
    csvData <- readFile "xor_train_data.csv"
    csvLabels <- readFile "xor_train_labels.csv"
    let dataMatrices = fromCsv csvData
        labelsMatrices = fromCsv csvLabels
        dataset = DataMap dataMatrices labelsMatrices
    sequence $ map print dataMatrices
    sequence $ map print labelsMatrices
