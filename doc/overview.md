<a name="nn.overview.dok"></a>
# Overview #

Each module of a network is composed of [Modules](module.md#nn.Modules) and there
are several sub-classes of `Module` available: container classes like
[Sequential](containers.md#nn.Sequential), [Parallel](containers.md#nn.Parallel) and
[Concat](containers.md#nn.Concat) , which can contain simple layers like
[Linear](simple.md#nn.Linear), [Mean](simple.md#nn.Mean), [Max](simple.md#nn.Max) and
[Reshape](simple.md#nn.Reshape), as well as [convolutional layers](convolution.md), and [transfer
functions](transfer.md) like [Tanh](transfer.md#nn.Tanh).

Loss functions are implemented as sub-classes of
[Criterion](criterion.md#nn.Criterions). They are helpful to train neural network on
classical tasks.  Common criterions are the Mean Squared Error
criterion implemented in [MSECriterion](criterion.md#nn.MSECriterion) and the
cross-entropy criterion implemented in
[ClassNLLCriterion](criterion.md#nn.ClassNLLCriterion).

Finally, the [StochasticGradient](training.md#nn.StochasticGradient) class provides a
high level way to train the neural network of choice, even though it is
easy with a simple for loop to [train a neural network yourself](training.md#nn.DoItYourself).

## Detailed Overview ##
This section provides a detailed overview of the neural network package. First the omnipresent [Module](#nn.overview.module) is examined, followed by some examples for [combining modules](#nn.overview.plugandplay) together. The last part explores facilities for [training a neural network](#nn.overview.training), and finally some caveats while training networks with [shared parameters](#nn.overview.sharedparams).

<a name="nn.overview.module"></a>
### Module ###

A neural network is called a [Module](module.md#nn.Module) (or simply
_module_ in this documentation) in Torch. `Module` is an abstract
class which defines four main methods:

  * [forward(input)](module.md#nn.Module.forward) which computes the output of the module given the `input` [Tensor](https://github.com/torch/torch7/blob/master/doc/tensor.md).
  * [backward(input, gradOutput)](module.md#nn.Module.backward) which computes the gradients of the module with respect to its own parameters, and its own inputs.
  * [zeroGradParameters()](module.md#nn.Module.zeroGradParameters) which zeroes the gradient with respect to the parameters of the module.
  * [updateParameters(learningRate)](module.md#nn.Module.updateParameters) which updates the parameters after one has computed the gradients with `backward()`

It also declares two members:

  * [output](module.md#nn.Module.output) which is the output returned by `forward()`.
  * [gradInput](module.md#nn.Module.gradInput) which contains the gradients with respect to the input of the module, computed in a `backward()`.

Two other perhaps less used but handy methods are also defined:

  * [share(mlp,s1,s2,...,sn)](module.md#nn.Module.share) which makes this module share the parameters s1,..sn of the module `mlp`. This is useful if you want to have modules that share the same weights.
  * [clone(...)](module.md#nn.Module.clone) which produces a deep copy of (i.e. not just a pointer to) this Module, including the current state of its parameters (if any).

Some important remarks:

  * `output` contains only valid values after a [forward(input)](module.md#nn.Module.forward).
  * `gradInput` contains only valid values after a [backward(input, gradOutput)](module.md#nn.Module.backward).
  * [backward(input, gradOutput)](module.md#nn.Module.backward) uses certain computations obtained during [forward(input)](module.md#nn.Module.forward). You _must_ call `forward()` before calling a `backward()`, on the _same_ `input`, or your gradients are going to be incorrect!

<a name="nn.overview.plugandplay"></a>
### Plug and play ###

Building a simple neural network can be achieved by constructing an available layer.
A linear neural network (perceptron!) is built only in one line:
```lua
mlp = nn.Linear(10,1) -- perceptron with 10 inputs
```

More complex neural networks are easily built using container classes
[Sequential](containers.md#nn.Sequential) and [Concat](containers.md#nn.Concat). `Sequential` plugs
layer in a feed-forward fully connected manner. `Concat` concatenates in
one layer several modules: they take the same inputs, and their output is
concatenated.

Creating a one hidden-layer multi-layer perceptron is thus just as easy as:
```lua
mlp = nn.Sequential()
mlp:add( nn.Linear(10, 25) ) -- 10 input, 25 hidden units
mlp:add( nn.Tanh() ) -- some hyperbolic tangent transfer function
mlp:add( nn.Linear(25, 1) ) -- 1 output
```

Of course, `Sequential` and `Concat` can contains other
`Sequential` or `Concat`, allowing you to try the craziest neural
networks you ever dreamt of!

<a name="nn.overview.training"></a>
### Training a neural network ###

Once you built your neural network, you have to choose a particular
[Criterion](criterion.md#nn.Criterions) to train it. A criterion is a class which
describes the cost to be minimized during training.

You can then train the neural network by using the
[StochasticGradient](training.md#nn.StochasticGradient) class.

```lua
 criterion = nn.MSECriterion() -- Mean Squared Error criterion
 trainer = nn.StochasticGradient(mlp, criterion)
 trainer:train(dataset) -- train using some examples
```

StochasticGradient expect as a `dataset` an object which implements
the operator `dataset[index]` and implements the method
`dataset:size()`. The `size()` methods returns the number of
examples and `dataset[i]` has to return the i-th example.

An `example` has to be an object which implements the operator
`example[field]`, where `field` might take the value `1` (input
features) or `2` (corresponding label which will be given to the
criterion).  The input is usually a Tensor (except if you use special
kind of gradient modules, like [table layers](table.md#nn.TableLayers)). The
label type depends on the criterion.  For example, the
[MSECriterion](criterion.md#nn.MSECriterion) expect a Tensor, but the
[ClassNLLCriterion](criterion.md#nn.ClassNLLCriterion) expect an integer number (the
class).

Such a dataset is easily constructed by using Lua tables, but it could be
any `C` object for example, as long as required operators/methods
are implemented.  [See an example](containers.md#nn.DoItStochasticGradient).

`StochasticGradient` being written in `Lua`, it is extremely easy
to cut-and-paste it and create a variant to it adapted to your needs
(if the constraints of `StochasticGradient` do not satisfy you).

<a name="nn.overview.lowlevel"></a>
#### Low Level Training ####

If you want to program the `StochasticGradient` by hand, you
essentially need to control the use of forwards and backwards through
the network yourself.  For example, here is the code fragment one
would need to make a gradient step given an input `x`, a desired
output `y`, a network `mlp` and a given criterion `criterion`
and learning rate `learningRate`:

```lua
function gradUpdate(mlp, x, y, criterion, learningRate) 
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
end
```
For example, if you wish to use your own criterion you can simply replace 
`gradCriterion` with the gradient vector of your criterion of choice.

<a name="nn.overview.sharedparams"></a>
### A Note on Sharing Parameters ###

By using `:share(...)` and the Container Modules, one can easily create very
complex architectures. In order to make sure that the network is going to
train properly, one needs to pay attention to the way the sharing is applied,
because it might depend on the optimization procedure.

* If you are using an optimization algorithm that iterates over the modules
of your network (by calling `:updateParameters` for example), only the
parameters of the network should be shared.
* If you use the flattened parameter tensor to optimize the network, 
obtained by calling `:getParameters`, for example for the package `optim`, 
then you need to share both the parameters and the gradParameters.

Here is an example for the first case:

```lua
-- our optimization procedure will iterate over the modules, so only share
-- the parameters
mlp = nn.Sequential()
linear = nn.Linear(2,2)
linear_clone = linear:clone('weight','bias') -- clone sharing the parameters
mlp:add(linear)
mlp:add(linear_clone)
function gradUpdate(mlp, x, y, criterion, learningRate) 
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
end
```

And for the second case:

```lua
-- our optimization procedure will use all the parameters at once, because
-- it requires the flattened parameters and gradParameters Tensors. Thus,
-- we need to share both the parameters and the gradParameters
mlp = nn.Sequential()
linear = nn.Linear(2,2)
-- need to share the parameters and the gradParameters as well
linear_clone = linear:clone('weight','bias','gradWeight','gradBias')
mlp:add(linear)
mlp:add(linear_clone)
params, gradParams = mlp:getParameters()
function gradUpdate(mlp, x, y, criterion, learningRate, params, gradParams)
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  -- adds the gradients to all the parameters at once
  params:add(-learningRate, gradParams)
end
```
