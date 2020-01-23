module LinReg
( coeffs
) where

import Data.Matrix (Matrix, transpose, inverse, fromLists,
    zero, nrows, (<|>))

-- coefficients of a linear regression of `y` on `xs`
coeffs :: (Eq t, Floating t) => Matrix t -> Matrix t -> Matrix t
coeffs xs y = either zeros (\a -> a * xs1_T * y) (inverse (xs1_T * xs1))
    where xs1_T = transpose xs1
          xs1 = ones <|> xs
          zeros = \_ -> zero nSamples 1
          ones = fromLists $ take nSamples (repeat [1])
          nSamples = nrows xs
