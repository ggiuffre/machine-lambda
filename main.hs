import qualified Data.Matrix as Mat
import Neural

-- orNet = Network [Mat.fromLists [[-1.0]]] [Mat.fromLists [[2.0, 2.0]]]
-- andNet = Network [Mat.fromLists [[-3.0]]] [Mat.fromLists [[2.0, 2.0]]]

b11 = Mat.fromLists [[-1], [-1]]
b12 = Mat.fromLists [[-1.25]]
w11 = Mat.fromLists [[2, -2], [-2, 2]]
w12 = Mat.fromLists [[2, 2]]
xorNet = Network [b11, b12] [w11, w12]

input = Mat.fromLists [[1], [1]]
label = Mat.fromLists [[0]]

states = analyze input xorNet
layers = zip (weights xorNet) (biases xorNet)
deltas = getErrors layers states label
as = input:(map snd states)
nablasW = gradientWeights as deltas
nablasB = gradientBiases as deltas

b21 = b11 - (head nablasB)
b22 = b12 - (head $ tail nablasB)
w21 = w11 - (head nablasW)
w22 = w12 - (head $ tail nablasW)
xor2 = Network [b21, b22] [w21, w22]

input = Mat.fromLists [[1], [0]]
label = Mat.fromLists [[1]]

states = analyze input xorNet
layers = zip (weights xorNet) (biases xorNet)
deltas = getErrors layers states label
as = input:(map snd states)
nablasW = gradientWeights as deltas
nablasB = gradientBiases as deltas

b31 = b21 - (head nablasB)
b32 = b22 - (head $ tail nablasB)
w31 = w21 - (head nablasW)
w32 = w22 - (head $ tail nablasW)
xor3 = Network [b31, b32] [w31, w32]
