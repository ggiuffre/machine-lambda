import qualified Data.Matrix as Mat
import Neural

-- orNet = Network [Mat.fromLists [[-1.0]]] [Mat.fromLists [[2.0, 2.0]]]
-- andNet = Network [Mat.fromLists [[-3.0]]] [Mat.fromLists [[2.0, 2.0]]]
-- b11 = Mat.fromLists [[-1], [-1]]
-- b12 = Mat.fromLists [[-1.25]]
-- w11 = Mat.fromLists [[2, -2], [-2, 2]]
-- w12 = Mat.fromLists [[2, 2]]
-- xorNet = Network [b11, b12] [w11, w12]

b11 = Mat.fromLists [[-0.09], [0.05]]
b12 = Mat.fromLists [[0.09]]
w11 = Mat.fromLists [[0.05, -0.08], [-0.02, -0.03]]
w12 = Mat.fromLists [[0.1, 0.05]]
xor1 = Network [b11, b12] [w11, w12]

input1 = Mat.fromLists [[0], [0]]
input2 = Mat.fromLists [[0], [1]]
input3 = Mat.fromLists [[1], [0]]
input4 = Mat.fromLists [[1], [1]]
label1 = Mat.fromLists [[0]]
label2 = Mat.fromLists [[1]]
label3 = Mat.fromLists [[1]]
label4 = Mat.fromLists [[0]]

states = analyze input1 xor1
layers = zip (weights xor1) (biases xor1)
deltas = getErrors layers states label1
as = input1:(map snd states)
nablasW = gradientWeights as deltas
nablasB = gradientBiases as deltas

b21 = b11 - (Mat.scaleMatrix 0.4 (head nablasB))
b22 = b12 - (Mat.scaleMatrix 0.4 (head $ tail nablasB))
w21 = w11 - (Mat.scaleMatrix 0.4 (head nablasW))
w22 = w12 - (Mat.scaleMatrix 0.4 (head $ tail nablasW))
xor2 = Network [b21, b22] [w21, w22]

states = analyze input2 xor2
layers = zip (weights xor2) (biases xor2)
deltas = getErrors layers states label2
as = input2:(map snd states)
nablasW = gradientWeights as deltas
nablasB = gradientBiases as deltas

b31 = b21 - (Mat.scaleMatrix 0.4 (head nablasB))
b32 = b22 - (Mat.scaleMatrix 0.4 (head $ tail nablasB))
w31 = w21 - (Mat.scaleMatrix 0.4 (head nablasW))
w32 = w22 - (Mat.scaleMatrix 0.4 (head $ tail nablasW))
xor3 = Network [b31, b32] [w31, w32]

states = analyze input3 xor3
layers = zip (weights xor3) (biases xor3)
deltas = getErrors layers states label3
as = input3:(map snd states)
nablasW = gradientWeights as deltas
nablasB = gradientBiases as deltas

b41 = b31 - (Mat.scaleMatrix 0.4 (head nablasB))
b42 = b32 - (Mat.scaleMatrix 0.4 (head $ tail nablasB))
w41 = w31 - (Mat.scaleMatrix 0.4 (head nablasW))
w42 = w32 - (Mat.scaleMatrix 0.4 (head $ tail nablasW))
xor4 = Network [b41, b42] [w41, w42]

states = analyze input4 xor4
layers = zip (weights xor4) (biases xor4)
deltas = getErrors layers states label4
as = input4:(map snd states)
nablasW = gradientWeights as deltas
nablasB = gradientBiases as deltas

b51 = b41 - (Mat.scaleMatrix 0.4 (head nablasB))
b52 = b42 - (Mat.scaleMatrix 0.4 (head $ tail nablasB))
w51 = w41 - (Mat.scaleMatrix 0.4 (head nablasW))
w52 = w42 - (Mat.scaleMatrix 0.4 (head $ tail nablasW))
xor5 = Network [b51, b52] [w51, w52]
