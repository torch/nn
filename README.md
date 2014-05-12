[![Build Status](https://travis-ci.org/torch/nn.svg?branch=master)](https://travis-ci.org/torch/nn)

<a name="nn.dok"/>
# Neural Network Package #

This package provides an easy way to build and train simple or complex
neural networks. The documentation is divided into different sections:
 * [Module](doc/module.md#nn.Module) : an abstract class inherited by the following layers:
   * [Module Containers](doc/containers.md#nn.Containers) : `Modules` that encapsulate other `Modules`;
   * [Transfer functions](doc/transfer.md#nn.transfer.dok) : non-linear functions like Tanh and Sigmoid;
   * [Simple layers](doc/simple.md#nn.simplelayers.dok) : parameterized layers like Linear and SparseLinear, or Tensor manipulations like Copy and Mean;
   * [Table layers](doc/table.md#nn.TableLayers) : layers for manipulating tables;
   * Convolution layers : temporal (1D), and spatial (2D) convolutions; 
 * [Criterion](doc/criterions.md#nn.Criterions) : given an input and a target, they compute a gradient according to a given loss function;
 * [Training](doc/training.md#nn.traningneuralnet.dok) a neural network;
 * [Detailed Overview](doc/overview.md#nn.overview.dok) of the package.

## Overview ##
Each module of a network is composed of [Modules](doc/module.md#nn.Modules) and there
are several sub-classes of `Module` available: container classes like
[Sequential](doc/containers.md#nn.Sequential), [Parallel](doc/containers.md#nn.Parallel) and
[Concat](doc/containers.md#nn.Concat) , which can contain simple layers like
[Linear](doc/simple.md#nn.Linear), [Mean](doc/simple.md#nn.Mean), [Max](doc/simple.md#nn.Max) and
[Reshape](doc/simple.md#nn.Reshape), as well as [convolutional layers](doc/convolution.md), and [transfer
functions](doc/transfer.md) like [Tanh](doc/transfer.md#nn.Tanh).

Loss functions are implemented as sub-classes of
[Criterion](doc/criterion.md#nn.Criterions). They are helpful to train neural network on
classical tasks.  Common criterions are the Mean Squared Error
criterion implemented in [MSECriterion](doc/criterion.md#nn.MSECriterion) and the
cross-entropy criterion implemented in
[ClassNLLCriterion](doc/criterion.md#nn.ClassNLLCriterion).

Finally, the [StochasticGradient](doc/training.md#nn.StochasticGradient) class provides a
high level way to train the neural network of choice, even though it is
easy with a simple for loop to [train a neural network yourself](doc/training.md#nn.DoItYourself).

## Testing ##
For those who want to implement their own modules, we suggest using
the `nn.Jacobian` class for testing the derivatives of their class,
together with the [torch.Tester](https://github.com/torch/torch7/blob/master/doc/tester.md) class. The sources
of `nn` package contains sufficiently many examples of such tests.

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
>>>>>>> 736794be54042fe48afc421ae4930e3d3aa4696e





