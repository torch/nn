# The R-op
The R-op is an efficient method of calculating the matrix-vector product of the
Hessian of a model and any vector. It is often used to quickly
approximate the curvature in a given direction, to speed up
optimization, or to calculate the full Hessian by repeatedly multiply
with one-hot vectors.

## Example
Given the following function,
```
h = σ(Wx + b)
y = σ(Wh + b)
E = sum(y)
```
this code calculates the product of the Hessian with a vector of ones.
```lua
local model = nn.Sequential()
   :add(nn.Linear(2, 2))
   :add(nn.Sigmoid())
   :add(nn.Linear(2, 2))
   :add(nn.Sigmoid())

-- We must collect the parameters, which also creates storage to store the
-- vector we will multiply with the Hessian, and to store the result (which the
-- R-op applied to the parameters).
local parameters, _, rParameters, rGradParameters = model:getParameters()
parameters:fill(1)

-- Set rParameters to the vector you want to multiply the Hessian with
rParameters:fill(1)

-- First do the normal forward and backward-propagation
local input = torch.ones(2)
model:forward(input)

-- Here we assume that the sum of the output is the cost, so the gradient is a
-- tensor of ones.
model:backward(input, torch.ones(2))

-- We calculate the R-ops as we go forward
model:rForward(input)

-- Since we assumed the cost to be the sum of the outputs, the second
-- derivative is zero which means that R(dE/dy) is zero.
model:rBackward(input, torch.zeros(2), torch.ones(2), torch.zeros(2))

-- The R-op applied to the parameters now contains the Hessian times the value
-- of rParameters
print(rGradParameters)
```
For details on the R-op, please see Pearlmutter, Barak A. "[Fast exact
multiplication by the
Hessian.](http://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf)"
Neural computation 6.1 (1994): 147-160.
