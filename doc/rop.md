# The R-op
The R-op is an efficient method of calculating the matrix-vector product of the
Hessian of a model and any vector. It is often used to quickly
approximate the curvature in a given direction, to speed up
optimization, or to calculate the full Hessian by repeatedly multiply
with one-hot vectors.

## Example
We use MLP to illustrate how to use R-op, the following codes compute the product
of the Hessian with a random vector.

```lua
local input_size = 3
local target_size = 3
local hidden_size = 2
local mini_batch_size = 10

local input = torch.randn(mini_batch_size, input_size)
local target = torch.ceil(torch.rand(mini_batch_size)*target_size)

local model = nn.Sequential()
   model:add(nn.Linear(input_size, hidden_size))
   model:add(nn.Sigmoid())
   model:add(nn.Linear(hidden_size, target_size))
   model:add(nn.LogSoftMax())
--local criterion = nn.MSECriterion()
local criterion = nn.ClassNLLCriterion()

-- We must collect the parameters, which also creates storage to store the
-- vector we will multiply with the Hessian, and to store the result (which the
-- R-op applied to the parameters).
local parameters, gradParameters, rParameters, rGradParameters = model:getParameters()
parameters:randn(parameters:size()) 


-- Set rParameters to the vector you want to multiply the Hessian with
rParameters:randn(parameters:size())

-- First do the normal forward and backward-propagation
local pred = model:forward(input)
local obj = criterion:forward(pred, target)

local df_do = criterion:backward(pred, target)
model:backward(input, df_do)

-- We calculate the R-ops as we go forward
local r_pred = model:rForward(input)

local rGradOutput =  criterion:rBackward(r_pred, target)
model:rBackward(input, torch.zeros(mini_batch_size, input_size), df_do, rGradOutput)


-- The R-op applied to the parameters now contains the Hessian times the value
-- of rParameters
print(rGradParameters)
```
For details on the R-op, please see Pearlmutter, Barak A. "[Fast exact
multiplication by the
Hessian.](http://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf)"
Neural computation 6.1 (1994): 147-160.
