[![Build Status](https://travis-ci.org/torch/nn.svg?branch=master)](https://travis-ci.org/torch/nn)

<a name="nn.dok"/>
# Neural Network Package #

This package provides an easy way to build and train simple or complex
neural networks.

Each module of a network is composed of [Modules](#nn.Modules) and there
are several sub-classes of `Module` available: container classes like
[Sequential](#nn.Sequential), [Parallel](#nn.Parallel) and
[Concat](#nn.Concat) , which can contain simple layers like
[Linear](#nn.Linear), [Mean](#nn.Mean), [Max](#nn.Max) and
[Reshape](#nn.Reshape), as well as convolutional layers, and transfer
functions like [Tanh](#nn.Tanh).

Loss functions are implemented as sub-classes of
[Criterion](#nn.Criterions). They are helpful to train neural network on
classical tasks.  Common criterions are the Mean Squared Error
criterion implemented in [MSECriterion](#nn.MSECriterion) and the
cross-entropy criterion implemented in
[ClassNLLCriterion](#nn.ClassNLLCriterion).

Finally, the [StochasticGradient](#nn.StochasticGradient) class provides a
high level way to train the neural network of choice, even though it is
easy with a simple for loop to [train a neural network yourself](#nn.DoItYourself).

For those who want to implement their own modules, we suggest using
the `nn.Jacobian` class for testing the derivatives of their class,
together with the [torch.Tester](..:torch:tester) class. The sources
of `nn` package contains sufficiently many examples of such tests.


<a name="nn.overview.dok"/>
# Detailed Overview of the Neural Network Package #

__Module__

A neural network is called a [Module](#nn.Module) (or simply
_module_ in this documentation) in Torch. `Module` is an abstract
class which defines four main methods:
  * [forward(input)](#nn.Module.forward) which computes the output of the module given the `input` [Tensor](..:torch:tensor).
  * [backward(input, gradOutput)](#nn.Module.backward) which computes the gradients of the module with respect to its own parameters, and its own inputs.
  * [zeroGradParameters()](#nn.Module.zeroGradParameters) which zeroes the gradient with respect to the parameters of the module.
  * [updateParameters(learningRate)](#nn.Module.updateParameters) which updates the parameters after one has computed the gradients with `backward()`

It also declares two members:
  * [output](#nn.Module.output) which is the output returned by `forward()`.
  * [gradInput](#nn.Module.gradInput) which contains the gradients with respect to the input of the module, computed in a `backward()`.

Two other perhaps less used but handy methods are also defined:
  * [share(mlp,s1,s2,...,sn)](#nn.Module.share) which makes this module share the parameters s1,..sn of the module `mlp`. This is useful if you want to have modules that share the same weights.
  * [clone(...)](#nn.Module.clone) which produces a deep copy of (i.e. not just a pointer to) this Module, including the current state of its parameters (if any).

Some important remarks:
  * `output` contains only valid values after a [forward(input)](#nn.Module.forward).
  * `gradInput` contains only valid values after a [backward(input, gradOutput)](#nn.Module.backward).
  * [backward(input, gradOutput)](#nn.Module.backward) uses certain computations obtained during [forward(input)](#nn.Module.forward). You _must_ call `forward()` before calling a `backward()`, on the _same_ `input`, or your gradients are going to be incorrect!


__Plug and play__

Building a simple neural network can be achieved by constructing an available layer.
A linear neural network (perceptron!) is built only in one line:
```lua
mlp = nn.Linear(10,1) -- perceptron with 10 inputs
```

More complex neural networks are easily built using container classes
[Sequential](#nn.Sequential) and [Concat](#nn.Concat). `Sequential` plugs
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
networks you ever dreamt of! See the [[#nn.Modules|complete list of
available modules]].

__Training a neural network__

Once you built your neural network, you have to choose a particular
[Criterion](#nn.Criterions) to train it. A criterion is a class which
describes the cost to be minimized during training.

You can then train the neural network by using the
[StochasticGradient](#nn.StochasticGradient) class.

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
kind of gradient modules, like [table layers](#nn.TableLayers)). The
label type depends of the criterion.  For example, the
[MSECriterion](#nn.MSECriterion) expect a Tensor, but the
[ClassNLLCriterion](#nn.ClassNLLCriterion) except a integer number (the
class).

Such a dataset is easily constructed by using Lua tables, but it could
any `C` object for example, as long as required operators/methods
are implemented.  [See an example](#nn.DoItStochasticGradient).

`StochasticGradient` being written in `Lua`, it is extremely easy
to cut-and-paste it and create a variant to it adapted to your needs
(if the constraints of `StochasticGradient` do not satisfy you).

__Low Level Training Of a Neural Network__

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
For example, if you wish to use your own criterion you can simple replace 
`gradCriterion` with the gradient vector of your criterion of choice.


<a name="nn.Modules"/>
# Modules #

Modules are bricks to build neural networks. A [Module](#nn.Module) is a neural network
by itself, but it can be combined with other networks using [container classes](#nn.Containers) to create
complex neural networks.

<a name="nn.Module"/>
## Module ##

`Module` is an abstract class which defines fundamental methods necessary
for a training a neural network. Modules are [serializable](..:torch:file#torch.file.serialization).

Modules contain two states variables: [output](#nn.ModuleOutput) and
[gradInput](#nn.ModuleGradInput).

<a name="nn.Module.forward"/>
### [output] forward(input) ###

Takes an `input` object, and computes the corresponding `output` of the
module. In general `input` and `output` are
[Tensors](..:torch:tensor). However, some special sub-classes
like [table layers](#nn.TableLayers) might expect something else. Please,
refer to each module specification for further information.

After a `forward()`, the [ouput](#nn.ModuleOutput) state variable should
have been updated to the new value.

It is not advised to override this function. Instead, one should
implement [updateOutput(input)](#nn.Module.updateOutput)
function. The forward module in the abstract parent class
[Module](#nn.Module) will call `updateOutput(input)`.

<a name="nn.Module.backward"/>
### [gradInput] backward(input, gradOutput) ###

Performs a _backpropagation step_ through the module, with respect to the
given `input`.  In general this method makes the assumption
[forward(input)](#nn.Module.forward) has been called before, _with the same input_.
This is necessary for optimization reasons. If you do not respect
this rule, `backward()` will compute incorrect gradients.

In general `input` and `gradOutput`  and `gradInput` are
[Tensors](..:torch:tensor). However, some special sub-classes
like [table layers](#nn.TableLayers) might expect something else. Please,
refer to each module specification for further information.

A _backpropagation step_ consist in computing two kind of gradients
at `input` given `gradOutput` (gradients with respect to the
output of the module).  This function simply performs this task using
two function calls:

  - A function call to [updateGradInput(input, gradOutput)](#nn.Module.updateGradInput).
  - A function call to [accGradParameters(input,gradOutput)](#nn.Module.accGradParameters).

It is not advised to override this function call in custom classes. It
is better to override
[updateGradInput(input, gradOutput)](#nn.Module.updateGradInput) and
[accGradParameters(input, gradOutput)](#nn.Module.accGradParameters)
functions.

<a name="nn.Module.updateOutput"/>
### updateOutput(input) ###

Computes the output using the current parameter set of the class and
input. This function returns the result which is stored in the
[output](#nn.Module.output) field.

<a name="nn.Module.updateGradInput"/>
### updateGradInput(input, gradOutput) ###

Computing the gradient of the module with respect to its own
input. This is returned in `gradInput`. Also, the
[gradInput](#nn.Module.gradInput) state variable is updated
accordingly.

<a name="nn.Module.accGradParameters"/>
### accGradParameters(input, gradOutput) ###

Computing the gradient of the module with respect to its
ownparameters. Many modules do not perform this step as they do not
have any parameters. The state variable name for the parameters is
module dependent. The module is expected to _accumulate_ the
gradients with respect to the parameters in some variable.

Zeroing this accumulation is achieved with
[zeroGradParameters()](#nn.Module.zeroGradParameters) and updating
the parameters according to this accumulation is done with
[updateParameters()](#nn.Module.updateParameters).

<a name="nn.Module.zeroGradParameters"/>
### zeroGradParameters() ###

If the module has parameters, this will zero the accumulation of the
gradients with respect to these parameters, accumulated through
[accGradParameters(input, gradOutput)](#nn.Module.accGradParameters)
calls. Otherwise, it does nothing.

<a name="nn.Module.updateParameters"/>
### updateParameters(learningRate) ###

If the module has parameters, this will update these parameters, according
to the accumulation of the gradients with respect to these parameters,
accumulated through [backward()](#nn.Module.backward) calls.

The update is basically:
```lua
parameters = parameters - learningRate * gradients_wrt_parameters
```
If the module does not have parameters, it does nothing.

<a name="nn.Module.accUpdateGradParameters"/>
### accUpdateGradParameters(input, gradOutput, learningRate) ###

This is a convenience module that performs two functions at
once. Calculates and accumulates the gradients with respect to the
weights after mutltiplying with negative of the learning rate
`learningRate`. Performing these two operations at once is more
performance efficient and it might be advantageous in certain
situations.

Keep in mind that, this function uses a simple trick to achieve its
goal and it might not be valid for a custom module.

Also note that compared to accGradParameters(), the gradients are not retained 
for future use. 

```lua
function Module:accUpdateGradParameters(input, gradOutput, lr)
   local gradWeight = self.gradWeight
   local gradBias = self.gradBias
   self.gradWeight = self.weight
   self.gradBias = self.bias
   self:accGradParameters(input, gradOutput, -lr)
   self.gradWeight = gradWeight
   self.gradBias = gradBias
end
```

As it can be seen, the gradients are accumulated directly into
weights. This assumption may not be true for a module that computes a
nonlinear operation.

<a name="nn.Module.share"/>
### share(mlp,s1,s2,...,sn) ###

This function modifies the parameters of the module named
`s1`,..`sn` (if they exist) so that they are shared with (pointers
to) the parameters with the same names in the given module `mlp`.

The parameters have to be Tensors. This function is typically used if
you want to have modules that share the same weights or biases.

Note that this function if called on a [Container](#nn.Containers)
module will share the same parameters for all the contained modules as
well.

Example:
```lua

-- make an mlp
mlp1=nn.Sequential(); 
mlp1:add(nn.Linear(100,10));

-- make a second mlp
mlp2=nn.Sequential(); 
mlp2:add(nn.Linear(100,10)); 

-- the second mlp shares the bias of the first
mlp2:share(mlp1,'bias');

-- we change the bias of the first
mlp1:get(1).bias[1]=99;

-- and see that the second one's bias has also changed..
print(mlp2:get(1).bias[1])

```


<a name="nn.Module.clone"/>
### clone(mlp,...) ###

Creates a deep copy of (i.e. not just a pointer to) the module,
including the current state of its parameters (e.g. weight, biases
etc., if any).

If arguments are provided to the `clone(...)` function it also calls
[share(...)](#nn.Module.share) with those arguments on the cloned
module after creating it, hence making a deep copy of this module with
some shared parameters.

Example:
```lua
-- make an mlp
mlp1=nn.Sequential(); 
mlp1:add(nn.Linear(100,10));

-- make a copy that shares the weights and biases
mlp2=mlp1:clone('weight','bias');

-- we change the bias of the first mlp
mlp1:get(1).bias[1]=99;

-- and see that the second one's bias has also changed..
print(mlp2:get(1).bias[1])

```

<a name="nn.Module.type"/>
### type(type) ###

This function converts all the parameters of a module to the given
`type`. The `type` can be one of the types defined for
[torch.Tensor](..:torch:tensor).

<a name="nn.Module.float"/>
### float() ###

Convenience method for calling [module:type('torch.FloatTensor')](#nn.Module.type)

<a name="nn.Module.double"/>
### double() ###

Convenience method for calling [module:type('torch.DoubleTensor')](#nn.Module.type)

<a name="nn.Module.cuda"/>
### cuda() ###

Convenience method for calling [module:type('torch.CudaTensor')](#nn.Module.type)

<a name="nn.statevars.dok"/>
### State Variables ###

These state variables are useful objects if one wants to check the guts of
a `Module`. The object pointer is _never_ supposed to change. However, its
contents (including its size if it is a Tensor) are supposed to change.

In general state variables are
[Tensors](..:torch:tensor). However, some special sub-classes
like [table layers](#nn.TableLayers) contain something else. Please,
refer to each module specification for further information.

<a name="nn.Module.output"/>
#### output ####

This contains the output of the module, computed with the last call of
[forward(input)](#nn.Module.forward).

<a name="nn.Module.gradInput"/>
#### gradInput ####

This contains the gradients with respect to the inputs of the module, computed with the last call of
[updateGradInput(input, gradOutput)](#nn.Module.updateGradInput). 

### Parameters and gradients w.r.t parameters ###

Some modules contain parameters (the ones that we actually want to
train!). The name of these parameters, and gradients w.r.t these parameters
are module dependent.

<a name="nn.Module.parameters"/>
### [{weights}, {gradWeights}] parameters() ###

This function should returns two tables. One for the learnable
parameters `{weights}` and another for the gradients of the energy
wrt to the learnable parameters `{gradWeights}`.

Custom modules should override this function if they use learnable
parameters that are stored in tensors.

<a name="nn.Module.getParameters"/>
### [flatParameters, flatGradParameters] getParameters() ###

This function returns two tensors. One for the flattened learnable
parameters `flatParameters` and another for the gradients of the energy
wrt to the learnable parameters `flatGradParameters`.

Custom modules should not override this function. They should instead override [parameters(...)](#nn.Module.parameters) which is, in turn, called by the present function.

This function will go over all the weights and gradWeights and make them view into a single tensor (one for weights and one for gradWeights). Since the storage of every weight and gradWeight is changed, this function should be called only once on a given network.

<a name="nn.Containers"/>
## Containers ##

<a name="nn.Concat"/>
### Concat ###

```lua
module = nn.Concat(dim)
```
Concat concatenates the output of one layer of "parallel" modules along the
provided dimension `dim`: they take the same inputs, and their output is
concatenated.
```lua
mlp=nn.Concat(1);
mlp:add(nn.Linear(5,3))
mlp:add(nn.Linear(5,7))
print(mlp:forward(torch.randn(5)))
```
which gives the output:
```lua
 0.7486
 0.1349
 0.7924
-0.0371
-0.4794
 0.3044
-0.0835
-0.7928
 0.7856
-0.1815
[torch.Tensor of dimension 10]
```


<a name="nn.Sequential"/>
### Sequential ###

Sequential provides a means to plug layers together
in a feed-forward fully connected manner.

E.g. 
creating a one hidden-layer multi-layer perceptron is thus just as easy as:
```lua
mlp = nn.Sequential()
mlp:add( nn.Linear(10, 25) ) -- 10 input, 25 hidden units
mlp:add( nn.Tanh() ) -- some hyperbolic tangent transfer function
mlp:add( nn.Linear(25, 1) ) -- 1 output

print(mlp:forward(torch.randn(10)))
```
which gives the output:
```lua
-0.1815
[torch.Tensor of dimension 1]
```

<a name="nn.Parallel"/>
### Parallel ###

`module` = `Parallel(inputDimension,outputDimension)`

Creates a container module that applies its `ith` child module to the  `ith` slice of the input Tensor by using [select](..:torch:tensor#torch.tensor.select) 
on dimension `inputDimension`. It concatenates the results of its contained modules together along dimension `outputDimension`.

Example:
```lua
 mlp=nn.Parallel(2,1);     -- iterate over dimension 2 of input
 mlp:add(nn.Linear(10,3)); -- apply to first slice
 mlp:add(nn.Linear(10,2))  -- apply to first second slice
 print(mlp:forward(torch.randn(10,2)))
```
gives the output:
```lua
-0.5300
-1.1015
 0.7764
 0.2819
-0.6026
[torch.Tensor of dimension 5]
```

A more complicated example:
```lua

mlp=nn.Sequential();
c=nn.Parallel(1,2)
for i=1,10 do
 local t=nn.Sequential()
 t:add(nn.Linear(3,2))
 t:add(nn.Reshape(2,1))
 c:add(t)
end
mlp:add(c)

pred=mlp:forward(torch.randn(10,3))
print(pred)

for i=1,10000 do     -- Train for a few iterations
 x=torch.randn(10,3);
 y=torch.ones(2,10);
 pred=mlp:forward(x)

 criterion= nn.MSECriterion()
 local err=criterion:forward(pred,y)
 local gradCriterion = criterion:backward(pred,y);
 mlp:zeroGradParameters();
 mlp:backward(x, gradCriterion); 
 mlp:updateParameters(0.01);
 print(err)
end
```
<a name="nn.simplelayers.dok"/>
## Simple layers ##

<a name="nn.Linear"/>
### Linear ###

`module` = `Linear(inputDimension,outputDimension)`

Applies a linear transformation to the incoming data, i.e.  //y=
Ax+b//. The `input` tensor given in `forward(input)` must be
either a vector (1D tensor) or matrix (2D tensor). If the input is a
matrix, then each row is assumed to be an input sample of given batch.

You can create a layer in the following way:
```lua
 module= nn.Linear(10,5)  -- 10 inputs, 5 outputs
```
Usually this would be added to a network of some kind, e.g.:
```lua
 mlp = nn.Sequential();
 mlp:add(module)
```
The weights and biases (_A_ and _b_) can be viewed with:
```lua
 print(module.weight)
 print(module.bias)
```
The gradients for these weights can be seen with:
```lua
 print(module.gradWeight)
 print(module.gradBias)
```
As usual with `nn` modules,
applying the linear transformation is performed with:
```lua
 x=torch.Tensor(10) -- 10 inputs
 y=module:forward(x)
```

<a name="nn.SparseLinear"/>
### SparseLinear ###

`module` = `SparseLinear(inputDimension,outputDimension)`

Applies a linear transformation to the incoming sparse data, i.e.
_y= Ax+b_. The `input` tensor given in `forward(input)` must
be a sparse vector represented as 2D tensor of the form 
torch.Tensor(N, 2) where the pairs represent indices and values.
The SparseLinear layer is useful when the number of input 
dimensions is very large and the input data is sparse.

You can create a sparse linear layer in the following way:

```lua
 module= nn.SparseLinear(10000,2)  -- 10000 inputs, 2 outputs
```
The sparse linear module may be used as part of a larger network, 
and apart from the form of the input, 
[SparseLinear](#nn.SparseLinear) 
operates in exactly the same way as the [Linear](#nn.Linear) layer.

A sparse input vector may be created as so..
```lua

 x=torch.Tensor({{1, 0.1},{2, 0.3},{10, 0.3},{31, 0.2}})

 print(x)

  1.0000   0.1000
  2.0000   0.3000
 10.0000   0.3000
 31.0000   0.2000
[torch.Tensor of dimension 4x2]

```

The first column contains indices, the second column contains 
values in a a vector where all other elements are zeros. The 
indices should not exceed the stated dimesions of the input to the 
layer (10000 in the example).

<a name="nn.Abs"/>
### Abs ###

`module` = `Abs()`

`output = abs(input)`.

```lua
m=nn.Abs()
ii=torch.linspace(-5,5)
oo=m:forward(ii)
go=torch.ones(100)
gi=m:backward(ii,go)
gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
gnuplot.grid(true)
```

![](doc/abs.png)

### Add ###
![](anchor:nn.Add)

`module` = `Add(inputDimension,scalar)`

Applies a bias term to the incoming data, i.e.
_y_i= x_i + b_i,  or if _scalar=true_ then uses a single bias term,
_y_i= x_i + b. 

Example:
```lua
y=torch.Tensor(5);  
mlp=nn.Sequential()
mlp:add(nn.Add(5))

function gradUpdate(mlp, x, y, criterion, learningRate) 
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
  return err
end

for i=1,10000 do
 x=torch.rand(5)
 y:copy(x); 
 for i=1,5 do y[i]=y[i]+i; end
 err=gradUpdate(mlp,x,y,nn.MSECriterion(),0.01)
end
print(mlp:get(1).bias)
```
gives the output:
```lua
 1.0000
 2.0000
 3.0000
 4.0000
 5.0000
[torch.Tensor of dimension 5]
```
i.e. the network successfully learns the input _x_ has been shifted 
to produce the output _y_.


<a name="nn.Mul"/>
### Mul ###

`module` = `Mul(inputDimension)`

Applies a _single_ scaling factor to the incoming data, i.e.
_y= w x_, where _w_ is a scalar. 

Example:
```lua
y=torch.Tensor(5);  
mlp=nn.Sequential()
mlp:add(nn.Mul(5))

function gradUpdate(mlp, x, y, criterion, learningRate) 
  local pred = mlp:forward(x)
  local err = criterion:forward(pred,y)
  local gradCriterion = criterion:backward(pred,y);
  mlp:zeroGradParameters();
  mlp:backward(x, gradCriterion);
  mlp:updateParameters(learningRate);
  return err
end


for i=1,10000 do
 x=torch.rand(5)
 y:copy(x); y:mul(math.pi);
 err=gradUpdate(mlp,x,y,nn.MSECriterion(),0.01)
end
print(mlp:get(1).weight)
```
gives the output:
```lua
 3.1416
[torch.Tensor of dimension 1]
```
i.e. the network successfully learns the input `x` has been scaled by
pi.

### CMul ###
![](anchor:nn.CMul)

`module` = `CMul(inputDimension)`

Applies a component-wise multiplication to the incoming data, i.e.
`y_i` = `w_i` =x_i=. 

Example:
```lua
mlp=nn.Sequential()
mlp:add(nn.CMul(5))

y=torch.Tensor(5); 
sc=torch.Tensor(5); for i=1,5 do sc[i]=i; end -- scale input with this

function gradUpdate(mlp,x,y,criterion,learningRate) 
  local pred = mlp:forward(x)
  local err = criterion:forward(pred,y)
  local gradCriterion = criterion:backward(pred,y);
  mlp:zeroGradParameters();
  mlp:backward(x, gradCriterion);
  mlp:updateParameters(learningRate);
  return err
end

for i=1,10000 do
 x=torch.rand(5)
 y:copy(x); y:cmul(sc);
 err=gradUpdate(mlp,x,y,nn.MSECriterion(),0.01)
end
print(mlp:get(1).weight)
```
gives the output:
```lua
 1.0000
 2.0000
 3.0000
 4.0000
 5.0000
[torch.Tensor of dimension 5]
```
i.e. the network successfully learns the input _x_ has been scaled by
those scaling factors to produce the output _y_.


<a name="nn.Max"/>
### Max ###

`module` = `Max(dimension)`

Applies a max operation over dimension `dimension`.
Hence, if an `nxpxq` Tensor was given as input, and `dimension` = `2`
then an `nxq` matrix would be output.


<a name="nn.Min"/>
### Min ###

`module` = `Min(dimension)`

Applies a min operation over dimension `dimension`.
Hence, if an `nxpxq` Tensor was given as input, and `dimension` = `2`
then an `nxq` matrix would be output.


<a name="nn.Mean"/>
### Mean ###

`module` = `Mean(dimension)`

Applies a mean operation over dimension `dimension`.
Hence, if an `nxpxq` Tensor was given as input, and `dimension` = `2`
then an `nxq` matrix would be output.

<a name="nn.Sum"/>
### Sum ###

`module` = `Sum(dimension)`

Applies a sum operation over dimension `dimension`.
Hence, if an `nxpxq` Tensor was given as input, and `dimension` = `2`
then an `nxq` matrix would be output.


<a name="nn.Euclidean"/>
### Euclidean ###

`module` = `Euclidean(inputDimension,outputDimension)`

Outputs the Euclidean distance of the input to `outputDimension` centers,
i.e. this layer has the weights `c_i`, `i` = `1`,..,`outputDimension`, where
`c_i` are vectors of dimension `inputDimension`. Output dimension `j` is
`|| c_j - x ||`, where `x` is the input.

<a name="nn.WeightedEuclidean"/>
### WeightedEuclidean ###

`module` = `WeightedEuclidean(inputDimension,outputDimension)`

This module is similar to [Euclidian](#nn.Euclidian), but
additionally learns a separate diagonal covariance matrix across the
features of the input space for each center.


<a name="nn.Copy"/>
### Copy ###

`module` = `Copy(inputType,outputType)`

This layer copies the input to output with type casting from input
type from `inputType` to `outputType`.


<a name="nn.Narrow"/>
### Narrow ###

`module` = `Narrow(dimension, offset, length)`

Narrow is application of
[narrow](..:torch:tensor:#torch.Tensor.narrow) operation in a
module.

<a name="nn.Replicate"/>
### Replicate ###

`module` = `Replicate(nFeature)`

This class creates an output where the input is replicated
`nFeature` times along its first dimension. There is no memory
allocation or memory copy in this module. It sets the
[stride](..:torch:tensor#torch.Tensor.stride) along the first
dimension to zero.

```lua
torch> x=torch.linspace(1,5,5)
torch> =x
 1
 2
 3
 4
 5
[torch.DoubleTensor of dimension 5]

torch> m=nn.Replicate(3)
torch> o=m:forward(x)
torch> =o
 1  2  3  4  5
 1  2  3  4  5
 1  2  3  4  5
[torch.DoubleTensor of dimension 3x5]

torch> x:fill(13)
torch> =x
 13
 13
 13
 13
 13
[torch.DoubleTensor of dimension 5]

torch> =o
 13  13  13  13  13
 13  13  13  13  13
 13  13  13  13  13
[torch.DoubleTensor of dimension 3x5]

```


<a name="nn.Reshape"/>
### Reshape ###

`module` = `Reshape(dimension1, dimension2, ..)`

Reshapes an `nxpxqx..`  Tensor into a `dimension1xdimension2x...` Tensor,
taking the elements column-wise.

Example:
```lua
> x=torch.Tensor(4,4)
> for i=1,4 do
>  for j=1,4 do
>   x[i][j]=(i-1)*4+j;
>  end
> end
> print(x)

  1   2   3   4
  5   6   7   8
  9  10  11  12
 13  14  15  16
[torch.Tensor of dimension 4x4]

> print(nn.Reshape(2,8):forward(x))

  1   9   2  10   3  11   4  12
  5  13   6  14   7  15   8  16
[torch.Tensor of dimension 2x8]

> print(nn.Reshape(8,2):forward(x))

  1   3
  5   7
  9  11
 13  15
  2   4
  6   8
 10  12
 14  16
[torch.Tensor of dimension 8x2]

> print(nn.Reshape(16):forward(x))

  1
  5
  9
 13
  2
  6
 10
 14
  3
  7
 11
 15
  4
  8
 12
 16
[torch.Tensor of dimension 16]


```


<a name="nn.Select"/>
### Select ###

Selects a dimension and index of a  `nxpxqx..`  Tensor.

Example:
```lua
mlp=nn.Sequential();
mlp:add(nn.Select(1,3))

x=torch.randn(10,5)
print(x)
print(mlp:forward(x))
```
gives the output:
```lua
 0.9720 -0.0836  0.0831 -0.2059 -0.0871
 0.8750 -2.0432 -0.1295 -2.3932  0.8168
 0.0369  1.1633  0.6483  1.2862  0.6596
 0.1667 -0.5704 -0.7303  0.3697 -2.2941
 0.4794  2.0636  0.3502  0.3560 -0.5500
-0.1898 -1.1547  0.1145 -1.1399  0.1711
-1.5130  1.4445  0.2356 -0.5393 -0.6222
-0.6587  0.4314  1.1916 -1.4509  1.9400
 0.2733  1.0911  0.7667  0.4002  0.1646
 0.5804 -0.5333  1.1621  1.5683 -0.1978
[torch.Tensor of dimension 10x5]

 0.0369
 1.1633
 0.6483
 1.2862
 0.6596
[torch.Tensor of dimension 5]
```

This can be used in conjunction with [Concat](#nn.Concat)
to emulate the behavior 
of [Parallel](#nn.Parallel), or to select various parts of an input Tensor to 
perform operations on. Here is a fairly complicated example:
```lua

mlp=nn.Sequential();
c=nn.Concat(2) 
for i=1,10 do
 local t=nn.Sequential()
 t:add(nn.Select(1,i))
 t:add(nn.Linear(3,2)) 
 t:add(nn.Reshape(2,1))
 c:add(t)
end
mlp:add(c)

pred=mlp:forward(torch.randn(10,3))
print(pred)

for i=1,10000 do     -- Train for a few iterations
 x=torch.randn(10,3);
 y=torch.ones(2,10);
 pred=mlp:forward(x)

 criterion= nn.MSECriterion()
 err=criterion:forward(pred,y)
 gradCriterion = criterion:backward(pred,y);
 mlp:zeroGradParameters();
 mlp:backward(x, gradCriterion); 
 mlp:updateParameters(0.01);
 print(err)
end
```

<a name="nn.Exp"/>
### Exp ###

Applies the `exp` function element-wise to the input Tensor,
thus outputting a Tensor of the same dimension.
```lua
ii=torch.linspace(-2,2)
m=nn.Exp()
oo=m:forward(ii)
go=torch.ones(100)
gi=m:backward(ii,go)
gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
gnuplot.grid(true)
```
![](doc/exp.png)


<a name="nn.Square"/>
### Square ###

Takes the square of each element.

```lua
ii=torch.linspace(-5,5)
m=nn.Square()
oo=m:forward(ii)
go=torch.ones(100)
gi=m:backward(ii,go)
gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
gnuplot.grid(true)
```
![](doc/square.png)

<a name="nn.Sqrt"/>
### Sqrt ###

Takes the square root of each element.

```lua
ii=torch.linspace(0,5)
m=nn.Sqrt()
oo=m:forward(ii)
go=torch.ones(100)
gi=m:backward(ii,go)
gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
gnuplot.grid(true)
```
![](doc/sqrt.png)

<a name="nn.Power"/>
### Power ###

`module` = `Power(p)`

Raises each element to its `pth` power.

```lua
ii=torch.linspace(0,2)
m=nn.Power(1.25)
oo=m:forward(ii)
go=torch.ones(100)
gi=m:backward(ii,go)
gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
gnuplot.grid(true)
```
![](doc/power.png)

<a name="nn.transfer.dok"/>
## Transfer Function Layers ##

<a name="nn.HardTanh"/>
### HardTanh ###

Applies the `HardTanh` function element-wise to the input Tensor,
thus outputting a Tensor of the same dimension.

`HardTanh` is defined as:

  * `f(x)` = `1, if x >`  `1,`
  * `f(x)` = `-1, if x <`  `-1,`
  * `f(x)` = `x,` `otherwise.`

```lua
ii=torch.linspace(-2,2)
m=nn.HardTanh()
oo=m:forward(ii)
go=torch.ones(100)
gi=m:backward(ii,go)
gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
gnuplot.grid(true)
```
![](doc/htanh.png)


<a name="nn.HardShrink"/>
### HardShrink ###

`module = nn.HardShrink(lambda)`

Applies the hard shrinkage function element-wise to the input
[Tensor](..:torch:Tensor). The output is the same size as the input.

`HardShrinkage` operator is defined as:

  * `f(x) = x, if x > lambda`
  * `f(x) = -x, if < -lambda`
  * `f(x) = 0, otherwise`

```lua
ii=torch.linspace(-2,2)
m=nn.HardShrink(0.85)
oo=m:forward(ii)
go=torch.ones(100)
gi=m:backward(ii,go)
gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
gnuplot.grid(true)
```
![](doc/hshrink.png)

<a name="nn.SoftShrink"/>
### SoftShrink ###

`module = nn.SoftShrink(lambda)`

Applies the hard shrinkage function element-wise to the input
[Tensor](..:torch:Tensor). The output is the same size as the input.

`HardShrinkage` operator is defined as:

  * `f(x) = x-lambda, if x > lambda`
  * `f(x) = -x+lambda, if < -lambda`
  * `f(x) = 0, otherwise`

```lua
ii=torch.linspace(-2,2)
m=nn.SoftShrink(0.85)
oo=m:forward(ii)
go=torch.ones(100)
gi=m:backward(ii,go)
gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
gnuplot.grid(true)
```
![](doc/sshrink.png)


<a name="nn.SoftMax"/>
### SoftMax ###

Applies the `Softmax` function to an n-dimensional input Tensor,
rescaling them so that the elements of the n-dimensional output Tensor
lie in the range (0,1) and sum to 1. 

`Softmax` is defined as `f_i(x)` = `exp(x_i-shift) / sum_j exp(x_j-shift)`,
where `shift` = `max_i x_i`.


```lua
ii=torch.exp(torch.abs(torch.randn(10)))
m=nn.SoftMax()
oo=m:forward(ii)
gnuplot.plot({'Input',ii,'+-'},{'Output',oo,'+-'})
gnuplot.grid(true)
```
![](doc/softmax.png)

<a name="nn.SoftMin"/>
### SoftMin ###

Applies the `Softmin` function to an n-dimensional input Tensor,
rescaling them so that the elements of the n-dimensional output Tensor
lie in the range (0,1) and sum to 1. 

`Softmin` is defined as `f_i(x)` = `exp(-x_i-shift) / sum_j exp(-x_j-shift)`,
where `shift` = `max_i x_i`.


```lua
ii=torch.exp(torch.abs(torch.randn(10)))
m=nn.SoftMin()
oo=m:forward(ii)
gnuplot.plot({'Input',ii,'+-'},{'Output',oo,'+-'})
gnuplot.grid(true)
```
![](doc/softmin.png)

<a name="nn.SoftPlus"/>
### SoftPlus ###

Applies the `SoftPlus` function to an n-dimensioanl input Tensor.
Can be used to constrain the output of a machine to always be positive.

`SoftPlus` is defined as `f_i(x)` = `log(1 + exp(x_i)))`.

```lua
ii=torch.randn(10)
m=nn.SoftPlus()
oo=m:forward(ii)
go=torch.ones(10)
gi=m:backward(ii,go)
gnuplot.plot({'Input',ii,'+-'},{'Output',oo,'+-'},{'gradInput',gi,'+-'})
gnuplot.grid(true)
```
![](doc/softplus.png)

<a name="nn.SoftSign"/>
### SoftSign ###

Applies the `SoftSign` function to an n-dimensioanl input Tensor.

`SoftSign` is defined as `f_i(x) = x_i / (1+|x_i|)`

```lua
ii=torch.linspace(-5,5)
m=nn.SoftSign()
oo=m:forward(ii)
go=torch.ones(100)
gi=m:backward(ii,go)
gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
gnuplot.grid(true)
```
![](doc/softsign.png)

<a name="nn.LogSigmoid"/>
### LogSigmoid ###

Applies the `LogSigmoid` function to an n-dimensional input Tensor.

`LogSigmoid` is defined as `f_i(x)` = `log(1/(1+ exp(-x_i)))`.


```lua
ii=torch.randn(10)
m=nn.LogSigmoid()
oo=m:forward(ii)
go=torch.ones(10)
gi=m:backward(ii,go)
gnuplot.plot({'Input',ii,'+-'},{'Output',oo,'+-'},{'gradInput',gi,'+-'})
gnuplot.grid(true)
```
![](doc/logsigmoid.png)


<a name="nn.LogSoftMax"/>
### LogSoftMax ###

Applies the `LogSoftmax` function to an n-dimensional input Tensor.

`LogSoftmax` is defined as `f_i(x)` = `log(1/a exp(x_i))`,
where  `a` = `sum_j exp(x_j)`.

```lua
ii=torch.randn(10)
m=nn.LogSoftMax()
oo=m:forward(ii)
go=torch.ones(10)
gi=m:backward(ii,go)
gnuplot.plot({'Input',ii,'+-'},{'Output',oo,'+-'},{'gradInput',gi,'+-'})
gnuplot.grid(true)
```
![](doc/logsoftmax.png)

<a name="nn.Sigmoid"/>
### Sigmoid ###

Applies the `Sigmoid` function element-wise to the input Tensor,
thus outputting a Tensor of the same dimension.

`Sigmoid` is defined as `f(x)` = `1/(1+exp(-x))`.

```lua
ii=torch.linspace(-5,5)
m=nn.Sigmoid()
oo=m:forward(ii)
go=torch.ones(100)
gi=m:backward(ii,go)
gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
gnuplot.grid(true)
```
![](doc/sigmoid.png)

<a name="nn.Tanh"/>
### Tanh ###

Applies the `Tanh` function element-wise to the input Tensor,
thus outputting a Tensor of the same dimension.

```lua
ii=torch.linspace(-3,3)
m=nn.Tanh()
oo=m:forward(ii)
go=torch.ones(100)
gi=m:backward(ii,go)
gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
gnuplot.grid(true)
```
![](doc/tanh.png)

<a name="nn.convlayers.dok"/>
## Convolutional layers ##

SpatialConvolution and SpatialSubsampling apply to inputs with
two-dimensional relationships (e.g. images).  TemporalConvolution and
TemporalSubsampling apply to sequences with a one-dimensional
relationship (e.g. strings of some kind).

For spatial convolutional layers, the input is supposed to be 3D. The
first dimension is the number of features, the last two dimenstions
are spatial.

<a name="nn.SpatialConvolution"/>
### SpatialConvolution ###

```lua
module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH])
```

Applies a 2D convolution over an input image composed of several input planes. The `input` tensor in
`forward(input)` is expected to be a 3D tensor (`nInputPlane x height x width`).

The parameters are the following:
  * `nInputPlane`: The number of expected input planes in the image given into `forward()`.
  * `nOutputPlane`: The number of output planes the convolution layer will produce.
  * `kW`: The kernel width of the convolution
  * `kH`: The kernel height of the convolution
  * `dW`: The step of the convolution in the width dimension. Default is `1`.
  * `dH`: The step of the convolution in the height dimension. Default is `1`.

Note that depending of the size of your kernel, several (of the last)
columns or rows of the input image might be lost. It is up to the user to
add proper padding in images.

If the input image is a 3D tensor `nInputPlane x height x width`, the output image size
will be `nOutputPlane x owidth x oheight` where
```lua
owidth  = (width  - kW) / dW + 1
oheight = (height - kH) / dH + 1 .
```

The parameters of the convolution can be found in `self.weight` (Tensor of
size `nOutputPlane x nInputPlane x kH x kW`) and `self.bias` (Tensor of
size `nOutputPlane`). The corresponding gradients can be found in
`self.gradWeight` and `self.gradBias`.

The output value of the layer can be precisely described as:
```lua
output[i][j][k] = bias[k]
  + sum_l sum_{s=1}^kW sum_{t=1}^kH weight[s][t][l][k]
                                    * input[dW*(i-1)+s)][dH*(j-1)+t][l]
```

<a name="nn.VolumetricConvolution"/>
### VolumetricConvolution ###

```lua
module = nn.VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH [, dT, dW, dH])
```

Applies a 3D convolution over an input image composed of several input planes. The `input` tensor in
`forward(input)` is expected to be a 4D tensor (`nInputPlane x time x height x width`).

The parameters are the following:
  * `nInputPlane`: The number of expected input planes in the image given into `forward()`.
  * `nOutputPlane`: The number of output planes the convolution layer will produce.
  * `kT`: The kernel size of the convolution in time
  * `kW`: The kernel width of the convolution
  * `kH`: The kernel height of the convolution
  * `dT`: The step of the convolution in the time dimension. Default is `1`.
  * `dW`: The step of the convolution in the width dimension. Default is `1`.
  * `dH`: The step of the convolution in the height dimension. Default is `1`.

Note that depending of the size of your kernel, several (of the last)
columns or rows of the input image might be lost. It is up to the user to
add proper padding in images.

If the input image is a 4D tensor `nInputPlane x time x height x width`, the output image size
will be `nOutputPlane x otime x owidth x oheight` where
```lua
otime   = (time  - kT) / dT + 1
owidth  = (width  - kW) / dW + 1
oheight = (height - kH) / dH + 1 .
```

The parameters of the convolution can be found in `self.weight` (Tensor of
size `nOutputPlane x nInputPlane x kT x kH x kW`) and `self.bias` (Tensor of
size `nOutputPlane`). The corresponding gradients can be found in
`self.gradWeight` and `self.gradBias`.

<a name="nn.SpatialConvolutionMap"/>
### SpatialConvolutionMap ###

```lua
module = nn.SpatialConvolutionMap(connectionMatrix, kW, kH, [dW], [dH])
```

This class is a generalization of
[nn.SpatialConvolution](#nn.SpatialConvolution). It uses a geenric
connection table between input and output features. The
[nn.SpatialConvolution](#nn.SpatialConvolution) is equivalent to
using a [full connection table](#nn.tables.full). One can specify
different types of connection tables.

<a name="nn.tables.full"/>
#### Full Connection Table ####

`table = nn.tables.full(nin,nout)`

This is a precomputed table that specifies connections between every
input and output node.

<a name="nn.tables.onetoone"/>
#### One to One Connection Table ####

`table = nn.tables.oneToOne(n)`

This is a precomputed table that specifies a single connection to each
output node from corresponding input node.

<a name="nn.tables.random"/>
#### Random Connection Table ####

`table = nn.tables.random(nin,nout, nto)`

This table is randomly populated such that each output unit has
`nto` incoming connections. The algorihtm tries to assign uniform
number of outgoing connections to each input node if possible.

<a name="nn.SpatialLPPooling"/>
### SpatialLPPooling ###

```lua
module = nn.SpatialLPPooling(nInputPlane, pnorm, kW, kH, [dW], [dH])
```

Computes the `p` norm in a convolutional manner on a set of 2D input planes.

<a name="nn.SpatialMaxPooling"/>
### SpatialMaxPooling ###

```lua
module = nn.SpatialMaxPooling(kW, kH [, dW, dH])
```

Applies 2D max-pooling operation in `kWxkH` regions by step size
`dWxdH` steps. The number of output features is equal to the number of
input planes.

<a name="nn.VolumetricMaxPooling"/>
### VolumetricMaxPooling ###

```lua
module = nn.VolumetricMaxPooling(kT, kW, kH [, dT, dW, dH])
```

Applies 3D max-pooling operation in `kTxkWxkH` regions by step size
`dTxdWxdH` steps. The number of output features is equal to the number of
input planes.

<a name="nn.SpatialSubSampling"/>
### SpatialSubSampling ###

```lua
module = nn.SpatialSubSampling(nInputPlane, kW, kH, [dW], [dH])
```

Applies a 2D sub-sampling over an input image composed of several input planes. The `input` tensor in
`forward(input)` is expected to be a 3D tensor (`nInputPlane x height x width`). The number of output
planes will be the same as `nInputPlane`.

The parameters are the following:
  * `nInputPlane`: The number of expected input planes in the image given into `forward()`.
  * `kW`: The kernel width of the sub-sampling
  * `kH`: The kernel height of the sub-sampling
  * `dW`: The step of the sub-sampling in the width dimension. Default is `1`.
  * `dH`: The step of the sub-sampling in the height dimension. Default is `1`.

Note that depending of the size of your kernel, several (of the last)
columns or rows of the input image might be lost. It is up to the user to
add proper padding in images.

If the input image is a 3D tensor `nInputPlane x height x width`, the output image size
will be `nInputPlane x oheight x owidth` where
```lua
owidth  = (width  - kW) / dW + 1
oheight = (height - kH) / dH + 1 .
```

The parameters of the sub-sampling can be found in `self.weight` (Tensor of
size `nInputPlane`) and `self.bias` (Tensor of size `nInputPlane`). The
corresponding gradients can be found in `self.gradWeight` and
`self.gradBias`.

The output value of the layer can be precisely described as:
```lua
output[i][j][k] = bias[k]
  + weight[k] sum_{s=1}^kW sum_{t=1}^kH input[dW*(i-1)+s)][dH*(j-1)+t][k]
```

<a name="nn.SpatialZeroPadding"/>
### SpatialZeroPadding ###

```lua
module = nn.SpatialZeroPadding(padLeft, padRight, padTop, padBottom)
```

Each feature map of a given input is padded with specified number of
zeros. If padding values are negative, then input is cropped.

<a name="nn.SpatialSubtractiveNormalization"/>
### SpatialSubtractiveNormalization ###

```lua
module = nn.SpatialSubtractiveNormalization(ninputplane, kernel)
```

Applies a spatial subtraction operation on a series of 2D inputs using
`kernel` for computing the weighted average in a neighborhood. The
neighborhood is defined for a local spatial region that is the size as
kernel and across all features. For a an input image, since there is
only one feature, the region is only spatial. For an RGB image, the
weighted anerage is taken over RGB channels and a spatial region.

If the `kernel` is 1D, then it will be used for constructing and seperable
2D kernel. The operations will be much more efficient in this case.

The kernel is generally chosen as a gaussian when it is believed that
the correlation of two pixel locations decrease with increasing
distance. On the feature dimension, a uniform average is used since
the weighting across features is not known.

For this example we use an external package
[image](http://www.github.com/clementfarabet/lua---image/)

```lua
require 'image'
require 'nn'
lena = image.rgb2y(image.lena())
ker = torch.ones(11)
m=nn.SpatialSubtractiveNormalization(1,ker)
processed = m:forward(lena)
w1=image.display(lena)
w2=image.display(processed)
```
![](lena.jpg)![](lenap.jpg)

<a name="nn.TemporalConvolution"/>
### TemporalConvolution ###

```lua
module = nn.TemporalConvolution(inputFrameSize, outputFrameSize, kW, [dW])
```

Applies a 1D convolution over an input sequence composed of `nInputFrame` frames. The `input` tensor in
`forward(input)` is expected to be a 2D tensor (`nInputFrame x inputFrameSize`) or a 3D tensor (`nBatchFrame x nInputFrame x inputFrameSize`).

The parameters are the following:
  * `inputFrameSize`: The input frame size expected in sequences given into `forward()`.
  * `outputFrameSize`: The output frame size the convolution layer will produce.
  * `kW`: The kernel width of the convolution
  * `dW`: The step of the convolution. Default is `1`.

Note that depending of the size of your kernel, several (of the last)
frames of the sequence might be lost. It is up to the user to add proper padding frames in the input
sequences.

If the input sequence is a 2D tensor of dimension `nInputFrame x inputFrameSize`, the output sequence will be
`nOutputFrame x outputFrameSize` where
```lua
nOutputFrame = (nInputFrame - kW) / dW + 1
```

If the input sequence is a 3D tensor of dimension `nBatchFrame x nInputFrame x inputFrameSize`, the output sequence will be
`nBatchFrame x nOutputFrame x outputFrameSize`.

The parameters of the convolution can be found in `self.weight` (Tensor of
size `outputFrameSize x (inputFrameSize x kW) `) and `self.bias` (Tensor of
size `outputFrameSize`). The corresponding gradients can be found in
`self.gradWeight` and `self.gradBias`.

For a 2D input, the output value of the layer can be precisely described as:
```lua
output[t][i] = bias[i]
  + sum_j sum_{k=1}^kW weight[i][j][k]
                                * input[dW*(t-1)+k)][j]
```

Here is a simple example:

```lua
inp=5;  -- dimensionality of one sequence element 
outp=1; -- number of derived features for one sequence element
kw=1;   -- kernel only operates on one sequence element per step
dw=1;   -- we step once and go on to the next sequence element

mlp=nn.TemporalConvolution(inp,outp,kw,dw)

x=torch.rand(7,inp) -- a sequence of 7 elements
print(mlp:forward(x))
```
which gives:
```lua
-0.9109
-0.9872
-0.6808
-0.9403
-0.9680 
-0.6901 
-0.6387
[torch.Tensor of dimension 7x1]
```

This is equivalent to:
```lua
weights=torch.reshape(mlp.weight,inp) -- weights applied to all
bias= mlp.bias[1];
for i=1,x:size(1) do -- for each sequence element
  element= x[i]; -- features of ith sequence element
  print(element:dot(weights) + bias)
end
```
which gives:
```lua
-0.91094998687717
-0.98721705771773
-0.68075004276185
-0.94030132495887
-0.96798754116609
-0.69008470895581
-0.63871422284166
```

<a name="nn.TemporalMaxPooling"/>
### TemporalMaxPooling ###

```lua
module = nn.TemporalMaxPooling(kW, [dW])
```

Applies 1D max-pooling operation in `kW` regions by step size
`dW` steps. Input sequence composed of `nInputFrame` frames. The `input` tensor in
`forward(input)` is expected to be a 2D tensor (`nInputFrame x inputFrameSize`) 
or a 3D tensor (`nBatchFrame x nInputFrame x inputFrameSize`).

If the input sequence is a 2D tensor of dimension `nInputFrame x inputFrameSize`, the output sequence will be
`nOutputFrame x inputFrameSize` where
```lua
nOutputFrame = (nInputFrame - kW) / dW + 1
```

<a name="nn.TemporalSubSampling"/>
### TemporalSubSampling ###

```lua
module = nn.TemporalSubSampling(inputFrameSize, kW, [dW])
```

Applies a 1D sub-sampling over an input sequence composed of `nInputFrame` frames. The `input` tensor in
`forward(input)` is expected to be a 2D tensor (`nInputFrame x inputFrameSize`). The output frame size
will be the same as the input one (`inputFrameSize`).

The parameters are the following:
  * `inputFrameSize`: The input frame size expected in sequences given into `forward()`.
  * `kW`: The kernel width of the sub-sampling
  * `dW`: The step of the sub-sampling. Default is `1`.

Note that depending of the size of your kernel, several (of the last)
frames of the sequence might be lost. It is up to the user to add proper padding frames in the input
sequences.

If the input sequence is a 2D tensor `nInputFrame x inputFrameSize`, the output sequence will be
`inputFrameSize x nOutputFrame` where
```lua
nOutputFrame = (nInputFrame - kW) / dW + 1
```

The parameters of the sub-sampling can be found in `self.weight` (Tensor of
size `inputFrameSize`) and `self.bias` (Tensor of
size `inputFrameSize`). The corresponding gradients can be found in
`self.gradWeight` and `self.gradBias`.

The output value of the layer can be precisely described as:
```lua
output[i][t] = bias[i] + weight[i] * sum_{k=1}^kW input[i][dW*(t-1)+k)]
```

<a name="nn.LookupTable"/>
### LookupTable ###

```lua
module = nn.LookupTable(nIndex, sizes)
```
or
```lua
module = nn.LookupTable(nIndex, size1, [size2], [size3], ...)
```

This layer is a particular case of a convolution, where the width of the convolution would be `1`.
When calling `forward(input)`, it assumes `input` is a 1D or 2D tensor filled with indices. 
If the input is a matrix, then each row is assumed to be an input sample of given batch. Indices start
at `1` and can go up to `nIndex`. For each index, it outputs a corresponding `Tensor` of size
specified by `sizes` (a `LongStorage`) or `size1 x size2 x...`.

Given a 1D input, the output tensors are concatenated, 
generating a `n x size1 x size2 x ... x sizeN` tensor, where `n`
is the size of a 1D `input` tensor. 

Again with a 1D input, when only `size1` is provided, the `forward(input)` is equivalent to 
performing the following matrix-matrix multiplication in an efficient manner:
```lua
M P
```
where `M` is a 2D matrix `size1 x nIndex` containing the parameters of the lookup-table and
`P` is a 2D matrix, where each column vector `i` is a zero vector except at index `input[i]` where it is `1`.

1D example:
```lua
 -- a lookup table containing 10 tensors of size 3
 module = nn.LookupTable(10, 3) 

 input = torch.Tensor{1,2,1,10}
 print(module:forward(input))
```

Outputs something like:
```lua
-1.4415 -0.1001 -0.1708
-0.6945 -0.4350  0.7977
-1.4415 -0.1001 -0.1708
-0.0745  1.9275  1.0915
[torch.DoubleTensor of dimension 4x3]
```
Note that the first row vector is the same as the 3rd one!

Given a 2D input tensor of size `m x n`, the output is a `m x n x size1 x size2 x ... x sizeN` 
tensor, where `m` is the number of samples in 
the batch and `n` is the number of indices per sample.

2D example:
```lua
 -- a lookup table containing 10 tensors of size 3
 module = nn.LookupTable(10, 3) 

 -- a batch of 2 samples of 4 indices each
 input = torch.Tensor({{1,2,4,5},{4,3,2,10}})
 print(module:forward(input))
```

Outputs something like:
```lua
(1,.,.) = 
 -0.0570 -1.5354  1.8555
 -0.9067  1.3392  0.6275
  1.9662  0.4645 -0.8111
  0.1103  1.7811  1.5969

(2,.,.) = 
  1.9662  0.4645 -0.8111
  0.0026 -1.4547 -0.5154
 -0.9067  1.3392  0.6275
 -0.0193 -0.8641  0.7396
[torch.DoubleTensor of dimension 2x4x3]
```


<a name="nn.TableLayers"/>
## Layers for manipulating tables ##

This set of modules allows the manipulation of  Tables
through the layers of a neural network.
This allows one to build very rich architectures.

Table-based modules work by supporting forward and backward methods that can accept 
tables as inputs. It turns out that the usual [Sequential](#nn.Sequential) module can do this, so all that is needed is other child modules that take advantage of such tables.
```lua
mlp = nn.Sequential();
t={x,y,z}
pred=mlp:forward(t)
pred=mlp:forward{x,y,z}      -- This is equivalent to the line before
```

<a name="nn.ConcatTable"/>
### ConcatTable ###

ConcatTable is a container module that applies each member module to 
the same input Tensor.

Example:
```lua
mlp= nn.ConcatTable()
mlp:add(nn.Linear(5,2))
mlp:add(nn.Linear(5,3))

pred=mlp:forward(torch.randn(5));
for i,k in pairs(pred) do print(i,k); end
```
which gives the output:
```lua
1
-0.4073
 0.0110
[torch.Tensor of dimension 2]

2
 0.0027
-0.0598
-0.1189
[torch.Tensor of dimension 3] 
```

<a name="nn.ParallelTable"/>
### ParallelTable ###

ParallelTable is a container module that, in its `forward` method, applies the `ith` member module to the `ith` input, and outputs a table of the set of outputs. 

Example:
```lua
mlp= nn.ParallelTable()
mlp:add(nn.Linear(10,2))
mlp:add(nn.Linear(5,3))

x=torch.randn(10)
y=torch.rand(5)

pred=mlp:forward{x,y}
for i,k in pairs(pred) do print(i,k); end
```
which gives the output:
```lua
1
 0.0331
 0.7003
[torch.Tensor of dimension 2]

2
 0.0677
-0.1657
-0.7383
[torch.Tensor of dimension 3]
```

<a name="nn.SplitTable"/>
### SplitTable ###

`module` = `SplitTable(dimension)`

Creates a module that takes a Tensor as input and outputs several tables, splitting the Tensor along dimension `dimension`.

Example 1:
```lua
mlp=nn.SplitTable(2)
x=torch.randn(4,3)
pred=mlp:forward(x)
for i,k in pairs(pred) do print(i,k); end
```
gives the output:
```lua
1
 1.3885
 1.3295
 0.4281
-1.0171
[torch.Tensor of dimension 4]

2
-1.1565
-0.8556
-1.0717
-0.8316
[torch.Tensor of dimension 4]

3
-1.3678
-0.1709
-0.0191
-2.5871
[torch.Tensor of dimension 4]
```

Example 2:
```lua
mlp=nn.SplitTable(1)
pred=mlp:forward(torch.randn(10,3))
for i,k in pairs(pred) do print(i,k); end
```
gives the output:
```lua
1
 1.6114
 0.9038
 0.8419
[torch.Tensor of dimension 3]

2
 2.4742
 0.2208
 1.6043
[torch.Tensor of dimension 3]

3
 1.3415
 0.2984
 0.2260
[torch.Tensor of dimension 3]

4
 2.0889
 1.2309
 0.0983
[torch.Tensor of dimension 3]
```

A more complicated example:
```lua

mlp=nn.Sequential();       --Create a network that takes a Tensor as input
mlp:add(nn.SplitTable(2))
 c=nn.ParallelTable()      --The two Tensors go through two different Linear
 c:add(nn.Linear(10,3))	   --Layers in Parallel
 c:add(nn.Linear(10,7))
mlp:add(c)                 --Outputing a table with 2 elements
 p=nn.ParallelTable()      --These tables go through two more linear layers
 p:add(nn.Linear(3,2))	   -- separately.
 p:add(nn.Linear(7,1)) 
mlp:add(p) 
mlp:add(nn.JoinTable(1))   --Finally, the tables are joined together and output. 

pred=mlp:forward(torch.randn(10,2))
print(pred)

for i=1,100 do             -- A few steps of training such a network.. 
 x=torch.ones(10,2);
 y=torch.Tensor(3); y:copy(x:select(2,1,1):narrow(1,1,3))
 pred=mlp:forward(x)

 criterion= nn.MSECriterion()
 local err=criterion:forward(pred,y)
 local gradCriterion = criterion:backward(pred,y);
 mlp:zeroGradParameters();
 mlp:backward(x, gradCriterion); 
 mlp:updateParameters(0.05);

 print(err)
end
```

<a name="nn.JoinTable"/>
### JoinTable ###

`module` = `JoinTable(dimension)`

Creates a module that takes a list of Tensors as input and outputs a Tensor by joining them together along dimension `dimension`.

Example:
```lua
x=torch.randn(5,1)
y=torch.randn(5,1)
z=torch.randn(2,1)

print(nn.JoinTable(1):forward{x,y})
print(nn.JoinTable(2):forward{x,y})
print(nn.JoinTable(1):forward{x,z})
```
gives the output:
```lua
1.3965
 0.5146
-1.5244
-0.9540
 0.4256
 0.1575
 0.4491
 0.6580
 0.1784
-1.7362
 
 1.3965  0.1575
 0.5146  0.4491
-1.5244  0.6580
-0.9540  0.1784
 0.4256 -1.7362

 1.3965
 0.5146
-1.5244
-0.9540
 0.4256
-1.2660
 1.0869
[torch.Tensor of dimension 7x1]
```

A more complicated example:
```lua

mlp=nn.Sequential();       --Create a network that takes a Tensor as input
 c=nn.ConcatTable()        --The same Tensor goes through two different Linear
 c:add(nn.Linear(10,3))	   --Layers in Parallel
 c:add(nn.Linear(10,7))
mlp:add(c)                 --Outputing a table with 2 elements
 p=nn.ParallelTable()      --These tables go through two more linear layers
 p:add(nn.Linear(3,2))	   -- separately.
 p:add(nn.Linear(7,1)) 
mlp:add(p) 
mlp:add(nn.JoinTable(1))   --Finally, the tables are joined together and output. 

pred=mlp:forward(torch.randn(10))
print(pred)

for i=1,100 do             -- A few steps of training such a network.. 
 x=torch.ones(10);
 y=torch.Tensor(3); y:copy(x:narrow(1,1,3))
 pred=mlp:forward(x)

 criterion= nn.MSECriterion()
 local err=criterion:forward(pred,y)
 local gradCriterion = criterion:backward(pred,y);
 mlp:zeroGradParameters();
 mlp:backward(x, gradCriterion); 
 mlp:updateParameters(0.05);

 print(err)
end
```

<a name="nn.Identity"/>
### Identity ###

`module` = `Identity()`

Creates a module that returns whatever is input to it as output. 
This is useful when combined with the module 
[ParallelTable](#nn.ParallelTable)
in case you do not wish to do anything to one of the input Tensors.
Example:
```lua
mlp=nn.Identity()
print(mlp:forward(torch.ones(5,2)))
```
gives the output: 
```lua
 1  1
 1  1
 1  1
 1  1
 1  1
[torch.Tensor of dimension 5x2]
```

Here is a more useful example, where one can implement a network which also computes a Criterion using this module:
```lua 
pred_mlp=nn.Sequential(); -- A network that makes predictions given x.
pred_mlp:add(nn.Linear(5,4)) 
pred_mlp:add(nn.Linear(4,3)) 

xy_mlp=nn.ParallelTable();-- A network for predictions and for keeping the
xy_mlp:add(pred_mlp)      -- true label for comparison with a criterion
xy_mlp:add(nn.Identity()) -- by forwarding both x and y through the network.

mlp=nn.Sequential();     -- The main network that takes both x and y.
mlp:add(xy_mlp)		 -- It feeds x and y to parallel networks;
cr=nn.MSECriterion();
cr_wrap=nn.CriterionTable(cr)
mlp:add(cr_wrap)         -- and then applies the criterion.

for i=1,100 do 		 -- Do a few training iterations
  x=torch.ones(5);          -- Make input features.
  y=torch.Tensor(3); 
  y:copy(x:narrow(1,1,3)) -- Make output label.
  err=mlp:forward{x,y}    -- Forward both input and output.
  print(err)		 -- Print error from criterion.

  mlp:zeroGradParameters();  -- Do backprop... 
  mlp:backward({x, y} );   
  mlp:updateParameters(0.05); 
end
```

<a name="nn.PairwiseDistance"/>
### PairwiseDistance ###

`module` = `PairwiseDistance(p)` creates a module that takes a table of two vectors as input and outputs the distance between them using the `p`-norm. 

Example:
```lua
mlp_l1=nn.PairwiseDistance(1)
mlp_l2=nn.PairwiseDistance(2)
x=torch.Tensor(1,2,3) 
y=torch.Tensor(4,5,6)
print(mlp_l1:forward({x,y}))
print(mlp_l2:forward({x,y}))
```
gives the output:
```lua
 9
[torch.Tensor of dimension 1]

 5.1962
[torch.Tensor of dimension 1]
```

A more complicated example:
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

<a name="nn.DotProduct"/>
### DotProduct ###

`module` = `DotProduct()` creates a module that takes a table of two vectors as input and outputs the dot product between them.

Example:
```lua
mlp=nn.DotProduct()
x=torch.Tensor(1,2,3) 
y=torch.Tensor(4,5,6)
print(mlp:forward({x,y}))
```
gives the output:
```lua
 32
[torch.Tensor of dimension 1]
```


A more complicated example:
```lua

-- Train a ranking function so that mlp:forward({x,y},{x,z}) returns a number
-- which indicates whether x is better matched with y or z (larger score = better match), or vice versa.

mlp1=nn.Linear(5,10)
mlp2=mlp1:clone('weight','bias')

prl=nn.ParallelTable();
prl:add(mlp1); prl:add(mlp2)

mlp1=nn.Sequential()
mlp1:add(prl)
mlp1:add(nn.DotProduct())

mlp2=mlp1:clone('weight','bias')

mlp=nn.Sequential()
prla=nn.ParallelTable()
prla:add(mlp1)
prla:add(mlp2)
mlp:add(prla)

x=torch.rand(5); 
y=torch.rand(5)
z=torch.rand(5)


print(mlp1:forward{x,x})
print(mlp1:forward{x,y})
print(mlp1:forward{y,y})


crit=nn.MarginRankingCriterion(1); 

-- Use a typical generic gradient update function
function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
end

inp={{x,y},{x,z}}

math.randomseed(1)

-- make the pair x and y have a larger dot product than x and z

for i=1,100 do
   gradUpdate(mlp,inp,1,crit,0.05)
   o1=mlp1:forward{x,y}[1]; 
   o2=mlp2:forward{x,z}[1]; 
   o=crit:forward(mlp:forward{{x,y},{x,z}},1)
   print(o1,o2,o)
end

print "________________**"

-- make the pair x and z have a larger dot product than x and y

for i=1,100 do
   gradUpdate(mlp,inp,-1,crit,0.05)
   o1=mlp1:forward{x,y}[1]; 
   o2=mlp2:forward{x,z}[1]; 
   o=crit:forward(mlp:forward{{x,y},{x,z}},-1)
   print(o1,o2,o)
end
```


<a name="nn.CosineDistance"/>
### CosineDistance ###

`module` = `CosineDistance()` creates a module that takes a table of two vectors as input and outputs the cosine distance between them.

Example:
```lua
mlp=nn.CosineDistance()
x=torch.Tensor(1,2,3) 
y=torch.Tensor(4,5,6)
print(mlp:forward({x,y}))
```
gives the output:
```lua
 0.9746
[torch.Tensor of dimension 1]
```

A more complicated example:
```lua

-- imagine we have one network we are interested in, it is called "p1_mlp"
p1_mlp= nn.Sequential(); p1_mlp:add(nn.Linear(5,2))

-- But we want to push examples towards or away from each other
-- so we make another copy of it called p2_mlp
-- this *shares* the same weights via the set command, but has its own set of temporary gradient storage
-- that's why we create it again (so that the gradients of the pair don't wipe each other)
p2_mlp= p1_mlp:clone('weight','bias')

-- we make a parallel table that takes a pair of examples as input. they both go through the same (cloned) mlp
prl = nn.ParallelTable()
prl:add(p1_mlp)
prl:add(p2_mlp)

-- now we define our top level network that takes this parallel table and computes the cosine distance betweem
-- the pair of outputs
mlp= nn.Sequential()
mlp:add(prl)
mlp:add(nn.CosineDistance())


-- lets make two example vectors
x=torch.rand(5)
y=torch.rand(5)

-- Grad update function..
function gradUpdate(mlp, x, y, learningRate)
local pred = mlp:forward(x)
if pred[1]*y < 1 then
 gradCriterion=torch.Tensor(-y)
 mlp:zeroGradParameters()
 mlp:backward(x, gradCriterion)
 mlp:updateParameters(learningRate)
end
end

-- push the pair x and y together, the distance should get larger..
for i=1,1000 do
 gradUpdate(mlp,{x,y},1,0.1)
 if ((i%100)==0) then print(mlp:forward({x,y})[1]);end
end


-- pull apart the pair x and y, the distance should get smaller..

for i=1,1000 do
 gradUpdate(mlp,{x,y},-1,0.1)
 if ((i%100)==0) then print(mlp:forward({x,y})[1]);end
end
```



<a name="nn.CriterionTable"/>
### CriterionTable ###

`module` = `CriterionTable(criterion)`

Creates a module that wraps a Criterion module so that it can accept a Table of inputs. Typically the table would contain two elements: the input and output `x` and `y` that the Criterion compares.

Example:
```lua
mlp = nn.CriterionTable(nn.MSECriterion())
x=torch.randn(5)
y=torch.randn(5)
print(mlp:forward{x,x})
print(mlp:forward{x,y})
```
gives the output:
```lua
0
1.9028918413199
```

Here is a more complex example of embedding the criterion into a network:
```lua

function table.print(t)
 for i,k in pairs(t) do print(i,k); end
end
 
mlp=nn.Sequential();                          -- Create an mlp that takes input
  main_mlp=nn.Sequential();		      -- and output using ParallelTable      
  main_mlp:add(nn.Linear(5,4)) 
  main_mlp:add(nn.Linear(4,3))
 cmlp=nn.ParallelTable(); 
 cmlp:add(main_mlp)
 cmlp:add(nn.Identity())           
mlp:add(cmlp)
mlp:add(nn.CriterionTable(nn.MSECriterion())) -- Apply the Criterion

for i=1,20 do                                 -- Train for a few iterations
 x=torch.ones(5);
 y=torch.Tensor(3); y:copy(x:narrow(1,1,3))
 err=mlp:forward{x,y}                         -- Pass in both input and output
 print(err)

 mlp:zeroGradParameters();
 mlp:backward({x, y} );   
 mlp:updateParameters(0.05); 
end
```

<a name="nn.CAddTable"/>
### CAddTable ###

Takes a table of tensors and outputs summation of all tensors.

```lua
ii = {torch.ones(5),torch.ones(5)*2,torch.ones(5)*3}
=ii[1]
 1
 1
 1
 1
 1
[torch.DoubleTensor of dimension 5]

return ii[2]
 2
 2
 2
 2
 2
[torch.DoubleTensor of dimension 5]

return ii[3]
 3
 3
 3
 3
 3
[torch.DoubleTensor of dimension 5]

m=nn.CAddTable()
=m:forward(ii)
 6
 6
 6
 6
 6
[torch.DoubleTensor of dimension 5]
```


<a name="nn.CSubTable"/>
### CSubTable ###

Takes a table with two tensor and returns the component-wise
subtraction between them.

```lua
m=nn.CSubTable()
=m:forward({torch.ones(5)*2.2,torch.ones(5)})
 1.2000
 1.2000
 1.2000
 1.2000
 1.2000
[torch.DoubleTensor of dimension 5]
```

<a name="nn.CMulTable"/>
### CMulTable ###

Takes a table of tensors and outputs the multiplication of all of them.

```lua
ii = {torch.ones(5)*2,torch.ones(5)*3,torch.ones(5)*4}
m=nn.CMulTable()
=m:forward(ii)
 24
 24
 24
 24
 24
[torch.DoubleTensor of dimension 5]

```

<a name="nn.CDivTable"/>
### CDivTable ###

Takes a table with two tensor and returns the component-wise
division between them.

```lua
m=nn.CDivTable()
=m:forward({torch.ones(5)*2.2,torch.ones(5)*4.4})
 0.5000
 0.5000
 0.5000
 0.5000
 0.5000
[torch.DoubleTensor of dimension 5]
```

<a name="nn.Criterions"/>
# Criterions #

Criterions are helpful to train a neural network. Given an input and a
target, they compute a gradient according to a given loss
function. [AbsCriterion](#nn.AbsCriterion) and
[MSECriterion](#nn.MSECriterion) are perfect for regression problems, while
[ClassNLLCriterion](#nn.ClassNLLCriterion) is the criterion of choice when
dealing with classification.

Criterions are [serializable](..:torch:file#torch.file.serialization).

<a name="nn.Criterion"/>
## Criterion ##

This is an abstract class which declares methods defined in all criterions.
This class is [serializable](..:torch:file#torch.file.serialization).

<a name="nn.Criterion.forward"/>
### [output] forward(input, target) ###

Given an `input` and a `target`, compute the loss function associated to the criterion and return the
result. In general `input` and `target` are [tensors](..:torch:tensor), but some specific criterions
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
criterion = AbsCriterion()
```

Creates a criterion that
measures the mean absolute value between `n` elements in the input `x` 
and output `y`:

`loss(x,y)`  = `1/n \sum |x_i-y_i|`.

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
criterion = ClassNLLCriterion()
```

The negative log likelihood criterion. It is useful to train a classication
problem with `n` classes. The `input` given through a `forward()` is
expected to contain _log-probabilities_ of each class: `input` has to be a
1D tensor of size `n`. Obtaining log-probabilities in a neural network is
easily achieved by adding a [LogSoftMax](#nn.LogSoftMax) layer in the last
layer of your neural network.

This criterion expect a class index (1 to the number of class) as `target`
when calling [forward(input, target)](#nn.CriterionForward) and
[backward(input, target)](#nn.CriterionBackward).

The loss can be described as:
```lua
loss(x, class) = forward(x, class) = -x[class]
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

<a name="nn.MarginCriterion"/>
## MarginCriterion ##

```lua
criterion = MarginCriterion()
```

Creates a criterion that optimizes a two-class classification hinge loss (margin-based loss) between input `x`  (a Tensor of dimension 1) and output `y` (which is a scalar, either 1 or -1) :

```lua
loss(x,y) = forward(x,y) = max(0,m- y x).
```

`m` is the margin, which is by default 1.

```lua
criterion = MarginCriterion(marginValue)
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
criterion = MultiMarginCriterion()
```

Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss) between input `x`  (a Tensor of dimension 1) and output `y` (which is a target class index, 1 <= y <= x:size(1)) :

```lua
loss(x,y) = forward(x,y) = sum_i(max(0, 1 - (x[y] - x[i]))) / x:size(1)
```
where i = 1 to x:size(1) and i ~= y

<a name="nn.MSECriterion"/>
## MSECriterion ##

```lua
criterion = MSECriterion()
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
criterion = MultiCriterion()
```

This returns a Criterion which is a weighted sum of other Criterion. 
Criterions are added using the method:

`criterion:add(singleCriterion, weight)`

where `weight` is a scalar.


<a name="nn.HingeEmbeddingCriterion"/>
## HingeEmbeddingCriterion ##

```lua
criterion = HingeEmbeddingCriterion()
```

Creates a criterion that measures the loss given  an input
`x` which is a 1-dimensional vector and a label `y` (1 or -1).
This is usually used for measuring whether two inputs are similar
or dissimilar, e.g. using the L1 pairwise distance, 
and is typically used for
learning nonlinear embeddings or semi-supervised learning.

<verbatim> 
loss(x,y) = forward(x,y) = x, if y=1
= max(0,margin - x), if y=-1
</verbatim>

The `margin` has a default value of 1, or can be set in the constructor:
```lua
criterion = HingeEmbeddingCriterion(marginValue)
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
criterion = L1HingeEmbeddingCriterion(margin)
```

Creates a criterion that measures the loss given  an input
`x` = `{x1,x2}`, a table of two tensors, and a label `y` (1 or -1):
This is used for measuring whether two inputs are similar
or dissimilar, using the L1 distance, and is typically used for
learning nonlinear embeddings or semi-supervised learning.

<verbatim> 
loss(x,y) = forward(x,y) = ||x1-x2||_1, if y=1
= max(0,margin - ||x1-x2||_1), if y=-1
</verbatim>

The `margin` has a default value of 1, or can be set in the constructor:
```lua
criterion = L1HingeEmbeddingCriterion(marginValue)
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
<verbatim> 
loss(x,y) = forward(x,y) = 1-cos(x1, x2), if y=1
= max(0,cos(x1, x2)-margin), if y=-1
</verbatim>

<a name="nn.MarginRankingCriterion"/>

## BCECriterion ##
```lua
criterion = nn.BCECriterion()
```

Creates a criterion that measures the Binary Cross Entropy between the target and the output:

crossentropy(t,o) = -(t * log(o) + (1 - t) * log(1 - o))

This is used for measuring the error of a reconstruction in for example an auto-encoder.

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
<verbatim> 
loss(x,y) = forward(x,y) = max(0,-y*(x[1]-x[2])+margin)
</verbatim>

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

<a name="nn.traningneuralnet.dok"/>
# Training a neural network #

Training a neural network is easy with a [simple `for` loop](#nn.DoItYourself).
While doing your own loop provides great flexibility, you might
want sometimes a quick way of training neural
networks. [StochasticGradient](#nn.StochasticGradient), a simple class
which does the job for you is provided as standard.

<a name="nn.StochasticGradient.dok"/>
## StochasticGradient ##

`StochasticGradient` is a high-level class for training [neural networks](#nn.Module), using a stochastic gradient
algorithm. This class is [serializable](..:torch:file#torch.file.serialization).

<a name="nn.StochasticGradient"/>
### StochasticGradient(module, criterion) ###

Create a `StochasticGradient` class, using the given [Module](#nn.Module) and [Criterion](#nn.Criterion).
The class contains [several parameters](#nn.StochasticGradientParameters) you might want to set after initialization.

<a name="nn.StochasticGradientTrain"/>
### train(dataset) ###

Train the module and criterion given in the
[constructor](#nn.StochasticGradient) over `dataset`, using the
internal [parameters](#nn.StochasticGradientParameters).

StochasticGradient expect as a `dataset` an object which implements the operator
`dataset[index]` and implements the method `dataset:size()`. The `size()` methods
returns the number of examples and `dataset[i]` has to return the i-th example.

An `example` has to be an object which implements the operator
`example[field]`, where `field` might take the value `1` (input features)
or `2` (corresponding label which will be given to the criterion). 
The input is usually a Tensor (except if you use special kind of gradient modules,
like [table layers](#nn.TableLayers)). The label type depends of the criterion.
For example, the [MSECriterion](#nn.MSECriterion) expects a Tensor, but the
[ClassNLLCriterion](#nn.ClassNLLCriterion) except a integer number (the class).

Such a dataset is easily constructed by using Lua tables, but it could any `C` object
for example, as long as required operators/methods are implemented. 
[See an example](#nn.DoItStochasticGradient).

<a name="nn.StochasticGradientParameters"/>
### Parameters ###

`StochasticGradient` has several field which have an impact on a call to [train()](#nn.StochasticGradientTrain).

  * `learningRate`: This is the learning rate used during training. The update of the parameters will be `parameters = parameters - learningRate * parameters_gradient`. Default value is `0.01`.
  * `learningRateDecay`: The learning rate decay. If non-zero, the learning rate (note: the field learningRate will not change value) will be computed after each iteration (pass over the dataset) with: `current_learning_rate =learningRate / (1 + iteration * learningRateDecay)`
  * `maxIteration`: The maximum number of iteration (passes over the dataset). Default is `25`.
  * `shuffleIndices`: Boolean which says if the examples will be randomly sampled or not. Default is `true`. If `false`, the examples will be taken in the order of the dataset.
  * `hookExample`: A possible hook function which will be called (if non-nil) during training after each example forwarded and backwarded through the network. The function takes `(self, example)` as parameters. Default is `nil`.
  * `hookIteration`: A possible hook function which will be called (if non-nil) during training after a complete pass over the dataset. The function takes `(self, iteration)` as parameters. Default is `nil`.

<a name="nn.DoItStochasticGradient"/>
## Example of training using StochasticGradient ##

We show an example here on a classical XOR problem.

__Dataset__

We first need to create a dataset, following the conventions described in
[StochasticGradient](#nn.StochasticGradientTrain).
```lua
dataset={};
function dataset:size() return 100 end -- 100 examples
for i=1,dataset:size() do 
  local input = torch.randn(2);     -- normally distributed example in 2d
  local output = torch.Tensor(1);
  if input[1]*input[2]>0 then     -- calculate label for XOR function
    output[1] = -1;
  else
    output[1] = 1
  end
  dataset[i] = {input, output}
end
```

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

__Training__

We choose the Mean Squared Error criterion and train the beast.
```lua
criterion = nn.MSECriterion()  
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)
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

<a name="nn.DoItYourself"/>
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

