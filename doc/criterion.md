<a name="nn.Criterions"/>
# Criterions #

Criterions are helpful to train a neural network. Given an input and a
target, they compute a gradient according to a given loss
function. [AbsCriterion](#nn.AbsCriterion) and
[MSECriterion](#nn.MSECriterion) are perfect for regression problems, while
[ClassNLLCriterion](#nn.ClassNLLCriterion) is the criterion of choice when
dealing with classification.

Criterions are [serializable](https://github.com/torch/torch7/blob/master/doc/file.md#serialization-methods).

<a name="nn.Criterion"/>
## Criterion ##

This is an abstract class which declares methods defined in all criterions.
This class is [serializable](https://github.com/torch/torch7/blob/master/doc/file.md#serialization-methods).

<a name="nn.Criterion.forward"/>
### [output] forward(input, target) ###

Given an `input` and a `target`, compute the loss function associated to the criterion and return the
result. In general `input` and `target` are [tensors](https://github.com/torch/torch7/blob/master/doc/tensor.md), but some specific criterions
might require some other type of object.

The `output` returned should be a scalar in general.

The state variable [self.output](#nn.Criterion.output) should be updated after a call to `forward()`.

<a name="nn.Criterion.backward"/>
### [gradInput] backward(input, target) ###

Given an `input` and a `target`, compute the gradients of the loss function associated to the criterion and
return the result.In general `input`, `target` and `gradInput` are [tensors](..:torch:tensor), but some specific criterions
might require some other type of object.

The state variable [self.gradInput](#nn.Criterion.gradInput) should be updated after a call to `backward()`.

<a name="nn.Criterion.output"/>
### State variable: output ###

State variable which contains the result of the last [forward(input, target)](#nn.Criterion.forward) call.

<a name="nn.Criterion.gradInput"/>
### State variable: gradInput ###

State variable which contains the result of the last [backward(input, target)](#nn.Criterion.backward) call.

<a name="nn.AbsCriterion"/>
## AbsCriterion ##

```lua
criterion = nn.AbsCriterion()
```

Creates a criterion that
measures the mean absolute value between `n` elements in the input `x` 
and output `y`:

```lua
loss(x,y)  = 1/n \sum |x_i-y_i|
```

If `x` and `y` are `d`-dimensional Tensors with a total of `n` elements,
the sum operation still operates over all the elements, and divides by `n`.

The division by `n` can be avoided if one sets the internal variable `sizeAverage` to `false`:

```lua
criterion = nn.AbsCriterion()
criterion.sizeAverage = false
```

<a name="nn.ClassNLLCriterion"/>
## ClassNLLCriterion ##

```lua
criterion = nn.ClassNLLCriterion(weights)
```

The negative log likelihood criterion. It is useful to train a classication
problem with `n` classes. 
If provided, the optional argument `weights` should be a 1D Tensor assigning weight to each of the classes. This is particularly useful when you have an unbalanced training set.

The `input` given through a `forward()` is
expected to contain _log-probabilities_ of each class: `input` has to be a
1D tensor of size `n`. 
Obtaining log-probabilities in a neural network is
easily achieved by adding a [LogSoftMax](#nn.LogSoftMax) layer in the last
layer of your neural network.
This criterion expect a class index (1 to the number of class) as `target`
when calling [forward(input, target)](#nn.CriterionForward) and
[backward(input, target)](#nn.CriterionBackward).

The loss can be described as:

```lua
loss(x, class) = forward(x, class) = -x[class]
```
or in the case of the `weights` argument being specified:

```lua
loss(x, class) = forward(x, class) = -weights[class]*x[class]
```
The following is a code fragment showing how to make a gradient step 
given an input `x`, a desired output `y` (an integer `1` to `n`, 
in this case `n` = `2` classes), 
a network `mlp` and a learning rate `learningRate`:

```lua
function gradUpdate(mlp,x,y,learningRate)
   local criterion = nn.ClassNLLCriterion()
   pred = mlp:forward(x)
   local err = criterion:forward(pred, y); 
   mlp:zeroGradParameters();
   local t = criterion:backward(pred, y);
   mlp:backward(x, t);
   mlp:updateParameters(learningRate);
end
```

<a name="nn.DistKLDivCriterion"/>
## DistKLDivCriterion ##

```lua
criterion = nn.DistKLDivCriterion()
```

Kullback–Leibler divergence criterion.  KL divergence is a useful distance 
measure for continuous distributions and is often useful when performance
direct regression over the space of (discretely sampled) continuous output 
distributions.  As with ClassNLLCriterion, the `input` given through a 
`forward()` is expected to contain _log-probabilities_, however unlike
ClassNLLCriterion, `input` is not restricted to a 1D vector.

This criterion expect a `target` tensor of the same size as the `input`
tensor when calling [forward(input, target)](#nn.CriterionForward) and
[backward(input, target)](#nn.CriterionBackward).

The loss can be described as:

```lua
loss(x, target) = sum_{all i}(target_i * (log(target_i) - x_i))
```

<a name="nn.BCECriterion"/>
## BCECriterion

```lua
criterion = nn.BCECriterion()
```

Creates a criterion that measures the Binary Cross Entropy between the target and the output:

```lua
loss(t,o) = -(t * log(o) + (1 - t) * log(1 - o))
```

This is used for measuring the error of a reconstruction in for example an auto-encoder.

<a name="nn.MarginCriterion"/>
## MarginCriterion ##

```lua
criterion = nn.MarginCriterion()
```

Creates a criterion that optimizes a two-class classification hinge loss (margin-based loss) between input `x`  (a Tensor of dimension 1) and output `y` (which is a scalar, either 1 or -1) :

```lua
loss(x,y) = forward(x,y) = max(0,m- y x).
```

`m` is the margin, which is by default 1.

```lua
criterion = nn.MarginCriterion(marginValue)
```

sets a different value of `m`.

Example:

```lua
require "nn"

function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
end

mlp=nn.Sequential()
mlp:add(nn.Linear(5,1))

x1=torch.rand(5)
x2=torch.rand(5)
criterion=nn.MarginCriterion(1)

for i=1,1000 do
   gradUpdate(mlp,x1,1,criterion,0.01)
   gradUpdate(mlp,x2,-1,criterion,0.01)
end

print(mlp:forward(x1))
print(mlp:forward(x2))

print(criterion:forward(mlp:forward(x1),1))
print(criterion:forward(mlp:forward(x2),-1))
```

gives the output:

```lua
 1.0043
[torch.Tensor of dimension 1]


-1.0061
[torch.Tensor of dimension 1]

0
0
```
i.e. the mlp successfully separates the two data points such that they both have a margin of 1, and hence a loss of 0.

<a name="nn.MultiMarginCriterion"/>
## MultiMarginCriterion ##

```lua
criterion = nn.MultiMarginCriterion(p)
```

Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss) between input `x`  (a Tensor of dimension 1) and output `y` (which is a target class index, 1 <= y <= x:size(1)) :

```lua
loss(x,y) = forward(x,y) = sum_i(max(0, 1 - (x[y] - x[i]))^p) / x:size(1)
```
where i = 1 to x:size(1) and i ~= y

<a name="nn.MSECriterion"/>
## MSECriterion ##

```lua
criterion = nn.MSECriterion()
```

Creates a criterion that measures the mean squared error between `n` elements in the input `x` 
and output `y`:

```lua
loss(x,y) = forward(x,y) = 1/n \sum |x_i-y_i|^2 .
```

If `x` and `y` are `d`-dimensional Tensors with a total of `n` elements,
the sum operation still operates over all the elements, and divides by `n`. The two tensors must
have the same number of elements (but their sizes might be different...)

The division by `n` can be avoided if one sets the internal variable `sizeAverage` to `false`:

```lua
criterion = nn.MSECriterion()
criterion.sizeAverage = false
```

<a name="nn.MultiCriterion"/>
## MultiCriterion ##

```lua
criterion = nn.MultiCriterion()
```

This returns a Criterion which is a weighted sum of other Criterion. 
Criterions are added using the method:

```lua
criterion:add(singleCriterion, weight)
```

where `weight` is a scalar.


<a name="nn.HingeEmbeddingCriterion"/>
## HingeEmbeddingCriterion ##

```lua
criterion = nn.HingeEmbeddingCriterion()
```

Creates a criterion that measures the loss given  an input
`x` which is a 1-dimensional vector and a label `y` (1 or -1).
This is usually used for measuring whether two inputs are similar
or dissimilar, e.g. using the L1 pairwise distance, 
and is typically used for
learning nonlinear embeddings or semi-supervised learning.

```
            ⎧ forward(x,y) = x,  if y=1
loss(x,y) = ⎨
            ⎩ max(0,margin - x), if y=-1
```

The `margin` has a default value of 1, or can be set in the constructor:

```lua
criterion = nn.HingeEmbeddingCriterion(marginValue)
```

Example use:
```lua
-- imagine we have one network we are interested in, it is called "p1_mlp"
p1_mlp= nn.Sequential(); p1_mlp:add(nn.Linear(5,2))

-- But we want to push examples towards or away from each other
-- so we make another copy of it called p2_mlp
-- this *shares* the same weights via the set command, but has its own set of temporary gradient storage
-- that's why we create it again (so that the gradients of the pair don't wipe each other)
p2_mlp= nn.Sequential(); p2_mlp:add(nn.Linear(5,2))
p2_mlp:get(1).weight:set(p1_mlp:get(1).weight)
p2_mlp:get(1).bias:set(p1_mlp:get(1).bias)

-- we make a parallel table that takes a pair of examples as input. they both go through the same (cloned) mlp
prl = nn.ParallelTable()
prl:add(p1_mlp)
prl:add(p2_mlp)

-- now we define our top level network that takes this parallel table and computes the pairwise distance betweem
-- the pair of outputs
mlp= nn.Sequential()
mlp:add(prl)
mlp:add(nn.PairwiseDistance(1))

-- and a criterion for pushing together or pulling apart pairs
crit=nn.HingeEmbeddingCriterion(1)

-- lets make two example vectors
x=torch.rand(5)
y=torch.rand(5)


-- Use a typical generic gradient update function
function gradUpdate(mlp, x, y, criterion, learningRate)
local pred = mlp:forward(x)
local err = criterion:forward(pred, y)
local gradCriterion = criterion:backward(pred, y)
mlp:zeroGradParameters()
mlp:backward(x, gradCriterion)
mlp:updateParameters(learningRate)
end

-- push the pair x and y together, notice how then the distance between them given
-- by  print(mlp:forward({x,y})[1]) gets smaller
for i=1,10 do
gradUpdate(mlp,{x,y},1,crit,0.01)
print(mlp:forward({x,y})[1])
end


-- pull apart the pair x and y, notice how then the distance between them given
-- by  print(mlp:forward({x,y})[1]) gets larger

for i=1,10 do
gradUpdate(mlp,{x,y},-1,crit,0.01)
print(mlp:forward({x,y})[1])
end

```

<a name="nn.L1HingeEmbeddingCriterion"/>
## L1HingeEmbeddingCriterion ##

```lua
criterion = nn.L1HingeEmbeddingCriterion(margin)
```

Creates a criterion that measures the loss given  an input
`x` = `{x1,x2}`, a table of two tensors, and a label `y` (1 or -1):
This is used for measuring whether two inputs are similar
or dissimilar, using the L1 distance, and is typically used for
learning nonlinear embeddings or semi-supervised learning.

```
            ⎧ forward(x,y) = ||x1-x2||_1,  if y=1
loss(x,y) = ⎨
            ⎩ max(0,margin - ||x1-x2||_1), if y=-1
```

The `margin` has a default value of 1, or can be set in the constructor:

```lua
criterion = nn.L1HingeEmbeddingCriterion(marginValue)
```

<a name="nn.CosineEmbeddingCriterion"/>
## CosineEmbeddingCriterion ##

```lua
criterion = nn.CosineEmbeddingCriterion(margin)
```

Creates a criterion that measures the loss given  an input
`x` = `{x1,x2}`, a table of two tensors, and a label `y` (1 or -1):
This is used for measuring whether two inputs are similar
or dissimilar, using the cosine distance, and is typically used for
learning nonlinear embeddings or semi-supervised learning.

`margin` should be a number from -1 to 1, 0 to 0.5 is suggested.
Forward and Backward have to be used alternately. If `margin` is missing, the default value is 0.

The loss function is:
```lua
            ⎧ forward(x,y) = 1-cos(x1, x2), if y=1
loss(x,y) = ⎨
            ⎩ max(0,cos(x1, x2)-margin),    if y=-1
```

<a name="nn.MarginRankingCriterion"/>
## MarginRankingCriterion ##

```lua
criterion = nn.MarginRankingCriterion(margin)
```

Creates a criterion that measures the loss given  an input
`x` = `{x1,x2}`, a table of two Tensors of size 1 (they contain only scalars),
and a label `y` (1 or -1):

If `y` = `1` then it assumed the first input should be ranked higher (have a larger value) 
than the second input, and vice-versa for `y` = `-1`.

The loss function is:

```lua
loss(x,y) = forward(x,y) = max(0,-y*(x[1]-x[2])+margin)
```

Example:

```lua
p1_mlp= nn.Linear(5,2)
p2_mlp= p1_mlp:clone('weight','bias')

prl=nn.ParallelTable()
prl:add(p1_mlp)
prl:add(p2_mlp)
  
mlp1=nn.Sequential()
mlp1:add(prl)
mlp1:add(nn.DotProduct())
 
mlp2=mlp1:clone('weight','bias')

mlpa=nn.Sequential()
prla=nn.ParallelTable()
prla:add(mlp1)
prla:add(mlp2)
mlpa:add(prla)

crit=nn.MarginRankingCriterion(0.1)

x=torch.randn(5)
y=torch.randn(5)
z=torch.randn(5)

-- Use a typical generic gradient update function
function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
end

for i=1,100 do
   gradUpdate(mlpa,{{x,y},{x,z}},1,crit,0.01)
   if true then 
      o1=mlp1:forward{x,y}[1]; 
      o2=mlp2:forward{x,z}[1]; 
      o=crit:forward(mlpa:forward{{x,y},{x,z}},1)
      print(o1,o2,o)
   end
end

print "--"

for i=1,100 do
   gradUpdate(mlpa,{{x,y},{x,z}},-1,crit,0.01)
   if true then 
      o1=mlp1:forward{x,y}[1]; 
      o2=mlp2:forward{x,z}[1]; 
      o=crit:forward(mlpa:forward{{x,y},{x,z}},-1)
      print(o1,o2,o)
   end
end
```

<a name="nn.L1Penalty"/>
## L1Penalty ##

```lua
penalty = nn.L1Penalty(L1weight, sizeAverage)
```

L1Penalty is an inline module that in it's FPROP copies the input Tensor directly to the output, and computes an L1 loss of the latent state (input) and stores it in the module's `loss` field.  During BPROP: `gradInput = gradOutput + gradLoss`.

This module can be used in autoencoder architectures to apply L1 losses to internal latent state without having to use Identity and parallel containers to carry the internal code to an output criterion.

Example (sparse autoencoder, note: decoder should be normalized):

```lua
encoder = nn.Sequential() 
encoder:add(nn.Linear(3, 128))
encoder:add(nn.Threshold())
decoder = nn.Linear(128,3)

autoencoder = nn.Sequential()
autoencoder:add(encoder)
autoencoder:add(nn.L1Penalty(l1weight))
autoencoder:add(decoder)

criterion = nn.MSECriterion()  -- To measure reconstruction error
-- ...
```

