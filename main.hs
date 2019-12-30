import qualified Data.Matrix as Mat
import DeepNN

-- orNet = zip [Mat.fromLists [[-1.0]]] [Mat.fromLists [[2.0, 2.0]]]
-- andNet = zip [Mat.fromLists [[-3.0]]] [Mat.fromLists [[2.0, 2.0]]]

b11 = Mat.fromLists [[-0.15269358], [-0.41688649]]
b12 = Mat.fromLists [[0.26026584]]
w11 = Mat.fromLists [[1.54684159, -0.45733174], [-1.20085651, -0.11960667]]
w12 = Mat.fromLists [[-2.05798541, -1.58322432]]
-- b11 = Mat.fromLists [[-1], [-1]]
-- b12 = Mat.fromLists [[-1.25]]
-- w11 = Mat.fromLists [[2, -2], [-2, 2]]
-- w12 = Mat.fromLists [[2, 2]]
xor1 = zip [b11, b12] [w11, w12]

input1 = Mat.fromLists [[0.0], [0.0]]
input2 = Mat.fromLists [[0.0], [1.0]]
input3 = Mat.fromLists [[1.0], [0.0]]
input4 = Mat.fromLists [[1.0], [1.0]]
label1 = Mat.fromLists [[0.0]]
label2 = Mat.fromLists [[1.0]]
label3 = Mat.fromLists [[1.0]]
label4 = Mat.fromLists [[0.0]]

-- dataset = zip [input1, input2, input3, input4] [label1, label2, label3, label4]
dataset = zip [input1, input2, input3, input4, input3, input2, input1, input4, input3, input2, input4, input1, input1, input2, input3, input4, input3, input2, input1, input4, input3, input4, input2, input1, input3, input4, input3, input2] [label1, label2, label3, label4, label3, label2, label1, label4, label3, label2, label4, label1, label1, label2, label3, label4, label3, label2, label1, label4, label3, label4, label2, label1, label3, label4, label3, label2] -- temporary hack to replace true randomness...
