import System.IO
import System.Random (StdGen, getStdGen, random)
import qualified Data.Matrix as Mat
import Dataset
import DeepNN

-- orNet = zip [Mat.fromLists [[-1.0]]] [Mat.fromLists [[2.0, 2.0]]]
-- andNet = zip [Mat.fromLists [[-3.0]]] [Mat.fromLists [[2.0, 2.0]]]

b11 = Mat.fromLists [[-0.15269358], [-0.41688649]]
b12 = Mat.fromLists [[0.26026584]]
w11 = Mat.fromLists [[1.54684159, -0.45733174], [-1.20085651, -0.11960667]]
w12 = Mat.fromLists [[-2.05798541, -1.58322432]]
xor1 = zip [b11, b12] [w11, w12]



main = do
    csvData <- readFile "xor_train_data.csv"
    csvLabels <- readFile "xor_train_labels.csv"
    let dataMatrices = fromCsv csvData
        labelsMatrices = fromCsv csvLabels
        dataset = zip dataMatrices labelsMatrices
    randGen <- getStdGen
    let (randInt, seed) = random randGen :: (Int, StdGen)
        epochs = shuffSgdUpdates xor1 dataset 0.8 seed
        xor2 = last $ take 5000 epochs
    sequence $ map (\x -> print $ infer x xor2) dataMatrices
