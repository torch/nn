<a name="nn.traningneuralnet.dok"></a>
# Training a neural network #

Training a neural network is easy with a [simple `for` loop](#nn.DoItYourself).
While doing your own loop provides great flexibility, you might
want sometimes a quick way of training neural
networks. [optim](https://github.com/torch/optim) is the standard way of training Torch7 neural networks.

`optim` is a quite general optimizer, for minimizing any function that outputs a loss.  In our case, our
function will be the loss of our network, given an input, and a set of weights.  The goal of training 
a neural net is to
optimize the weights to give the lowest loss over our training set of input data.  So, we are going to use optim
to minimize the loss with respect to the weights, over our training set.  We will feed the data to 
`optim` in minibatches.  For this particular example, we will use just one minibatch, but in your own training
you will almost certainly want to break your training set into minibatches, and feed each minibatch to `optim`,
one by one.

We need to give `optim` a function that will output the loss and the derivative of the loss with respect to the
weights, given a set of input weights.  The function will have access to our training minibatch, and use this
to calculate the loss, for this minibatch.  Typically, the function would be defined inside our loop over
batches, and therefore have access to the current minibatch data.

Here's how this looks:

__Neural Network__

We create a simple neural network with one hidden layer.
```lua
require 'nn'

local model = nn.Sequential();  -- make a multi-layer perceptron
local inputs = 2; outputs = 1; HUs = 20; -- parameters
model:add(nn.Linear(inputs, HUs))
model:add(nn.Tanh())
model:add(nn.Linear(HUs, outputs))
```

__Criterion__

We choose the Mean Squared Error criterion and train the dataset.
```lua
local criterion = nn.MSECriterion()
```

__Dataset__

We will just create one minibatch of 128 examples.  In your own networks, you'd want to break down your
rather larger dataset into multiple minibatches, of around 32-512 examples each.

```
local batchSize = 128
local batchInputs = torch.Tensor(batchSize, inputs)
local batchLabels = torch.ByteTensor(batchSize)

for i=1,batchSize do
  local input = torch.randn(2)     -- normally distributed example in 2d
  local label = 1
  if input[1]*input[2]>0 then     -- calculate label for XOR function
    label = -1;
  end
  batchInputs[i]:copy(input)
  batchLabels[i] = label
end
```

__Training__

`optim` provides []various training algorithms](https://github.com/torch/optim/blob/master/doc/index.md).  We
will use [Stochastic Gradient Descent](https://github.com/torch/optim/blob/master/doc/index.md#x-sgdopfunc-x-state).  We
need to provide the learning rate, via an optimization state table:

```
require 'optim'

local optimState = {learningRate=0.01}

-- retrieve the weights and biases from the model, as 1-dimensional flattened tensors
-- these are views onto the underlying weights and biases, and we will give them to optim
-- When optim updates these params, it is implicitly updating the weights and biases of our
-- models
local params, gradParams = model:getParameters()
for epoch=1,50 do
  -- local function we give to optim
  -- it takes current weights as input, and outputs the loss
  -- and the gradient of the loss with respect to the weights
  -- gradParams is calculated implicitly by calling 'backward'
  -- because gradParams is a view onto the model's weight and bias
  -- gradients tensor
  local function feval(params)
    gradParams:zero()

    local outputs = model:forward(batchInputs)
    local loss = criterion:forward(outputs, batchLabels)
    local dloss_doutput = criterion:backward(outputs, batchLabels)
    model:backward(batchInputs, dloss_doutput)

    return loss,gradParams
  end
  optim.sgd(feval, params, optimState)
end
```
__Test the network__

```lua
x = torch.Tensor(2)
x[1] =  0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] =  0.5; x[2] = -0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))
```

You should see something like:
```lua
> x = torch.Tensor(2)
> x[1] =  0.5; x[2] =  0.5; print(mlp:forward(x))

-0.3490
[torch.Tensor of dimension 1]

> x[1] =  0.5; x[2] = -0.5; print(mlp:forward(x))

 1.0561
[torch.Tensor of dimension 1]

> x[1] = -0.5; x[2] =  0.5; print(mlp:forward(x))

 0.8640
[torch.Tensor of dimension 1]

> x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))

-0.2941
[torch.Tensor of dimension 1]
```

<a name="nn.DoItYourself"></a>
## Example of manual training of a neural network ##

We show an example here on a classical XOR problem.

__Neural Network__

We create a simple neural network with one hidden layer.
```lua
require "nn"
mlp = nn.Sequential();  -- make a multi-layer perceptron
inputs = 2; outputs = 1; HUs = 20; -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))
```

__Loss function__

We choose the Mean Squared Error criterion.
```lua
criterion = nn.MSECriterion()  
```

__Training__

We create data _on the fly_ and feed it to the neural network.

```lua
for i = 1,2500 do
  -- random sample
  local input= torch.randn(2);     -- normally distributed example in 2d
  local output= torch.Tensor(1);
  if input[1]*input[2] > 0 then  -- calculate label for XOR function
    output[1] = -1
  else
    output[1] = 1
  end

  -- feed it to the neural network and the criterion
  criterion:forward(mlp:forward(input), output)

  -- train over this example in 3 steps
  -- (1) zero the accumulation of the gradients
  mlp:zeroGradParameters()
  -- (2) accumulate gradients
  mlp:backward(input, criterion:backward(mlp.output, output))
  -- (3) update parameters with a 0.01 learning rate
  mlp:updateParameters(0.01)
end
```

__Test the network__

```lua
x = torch.Tensor(2)
x[1] =  0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] =  0.5; x[2] = -0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))
```

You should see something like:
```lua
> x = torch.Tensor(2)
> x[1] =  0.5; x[2] =  0.5; print(mlp:forward(x))

-0.6140
[torch.Tensor of dimension 1]

> x[1] =  0.5; x[2] = -0.5; print(mlp:forward(x))

 0.8878
[torch.Tensor of dimension 1]

> x[1] = -0.5; x[2] =  0.5; print(mlp:forward(x))

 0.8548
[torch.Tensor of dimension 1]

> x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))

-0.5498
[torch.Tensor of dimension 1]
```
