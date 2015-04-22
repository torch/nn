<a name="nn.transfer.dok"/>
# Transfer Function Layers #
Transfer functions are normally used to introduce a non-linearity after a parameterized layer like [Linear](simple.md#nn.Linear) and  [SpatialConvolution](convolution.md#nn.SpatialConvolution). Non-linearities allows for dividing the problem space into more complex regions than what a simple logistic regressor would permit.

<a name="nn.HardTanh"/>
## HardTanh ##

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
![](image/htanh.png)


<a name="nn.HardShrink"/>
## HardShrink ##

`module = nn.HardShrink(lambda)`

Applies the hard shrinkage function element-wise to the input
[Tensor](https://github.com/torch/torch7/blob/master/doc/tensor.md). The output is the same size as the input.

`HardShrinkage` operator is defined as:

  * `f(x) = x, if x > lambda`
  * `f(x) = -x, if x < -lambda`
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
![](image/hshrink.png)

<a name="nn.SoftShrink"/>
## SoftShrink ##

`module = nn.SoftShrink(lambda)`

Applies the hard shrinkage function element-wise to the input
[Tensor](https://github.com/torch/torch7/blob/master/doc/tensor.md). The output is the same size as the input.

`HardShrinkage` operator is defined as:

  * `f(x) = x-lambda, if x > lambda`
  * `f(x) = -x+lambda, if x < -lambda`
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
![](image/sshrink.png)


<a name="nn.SoftMax"/>
## SoftMax ##

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
![](image/softmax.png)

Note that this module doesn't work directly with [ClassNLLCriterion](criterion.md#nn.ClassNLLCriterion), which expects the `nn.Log` to be computed between the `SoftMax` and itself. Use [LogSoftMax](#nn.LogSoftMax) instead (it's faster).

<a name="nn.SoftMin"/>
## SoftMin ##

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
![](image/softmin.png)

<a name="nn.SoftPlus"/>
### SoftPlus ###

Applies the `SoftPlus` function to an n-dimensioanl input Tensor.
Can be used to constrain the output of a machine to always be positive.

`SoftPlus` is defined as `f_i(x)` = `1/beta * log(1 + exp(beta * x_i))`.

```lua
ii=torch.randn(10)
m=nn.SoftPlus()
oo=m:forward(ii)
go=torch.ones(10)
gi=m:backward(ii,go)
gnuplot.plot({'Input',ii,'+-'},{'Output',oo,'+-'},{'gradInput',gi,'+-'})
gnuplot.grid(true)
```
![](image/softplus.png)

<a name="nn.SoftSign"/>
## SoftSign ##

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
![](image/softsign.png)

<a name="nn.LogSigmoid"/>
## LogSigmoid ##

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
![](image/logsigmoid.png)


<a name="nn.LogSoftMax"/>
## LogSoftMax ##

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
![](image/logsoftmax.png)

<a name="nn.Sigmoid"/>
## Sigmoid ##

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
![](image/sigmoid.png)

<a name="nn.Tanh"/>
## Tanh ##

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
![](image/tanh.png)

<a name="nn.ReLU"/>
## ReLU ##

Applies the rectified linear unit (`ReLU`) function element-wise to the input Tensor,
thus outputting a Tensor of the same dimension.

Can optionally do it's operation in-place without using extra state memory:
```lua
m=nn.ReLU(true) -- true = in-place, false = keeping separate state.
```

```lua
ii=torch.linspace(-3,3)
m=nn.ReLU()
oo=m:forward(ii)
go=torch.ones(100)
gi=m:backward(ii,go)
gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
gnuplot.grid(true)
```
![](image/relu.png)

<a name="nn.PReLU"/>
## PReLU ##

Applies parametric ReLU, which parameter varies the slope of the negative part:

```max(0,y_i) + a_i * min(0,y_i)```

When called without a number on input as ```nn.PReLU()``` uses shared version, meaning
has only one parameter. Otherwise if called ```nn.PReLU(nOutputPlane)``` has ```nOutputPlane```
parameters, one for each input map. The output dimension is always equal to input dimension.
Note that weight decay should not be used on it. For reference see http://arxiv.org/abs/1502.01852

![](image/prelu.png)

<a name="nn.AddConstant"/>
## AddConstant ##

Adds a (non-learnable) scalar constant.  This module is sometimes useful for debuggging purposes:  `f(x)` = `x + k`, where `k` is a scalar.

Can optionally do it's operation in-place without using extra state memory:
```lua
m=nn.AddConstant(k,true) -- true = in-place, false = keeping separate state.
```
In-place mode restores the original input value after the backward pass, allowing it's use after other in-place modules, like [MulConstant](#nn.MulConstant).

<a name="nn.MulConstant"/>
## MulConstant ##

Multiplies input tensor by a (non-learnable) scalar constant.  This module is sometimes useful for debuggging purposes:  `f(x)` = `k * x`, where `k` is a scalar.

Can optionally do it's operation in-place without using extra state memory:
```lua
m=nn.MulConstant(k,true) -- true = in-place, false = keeping separate state.
```
In-place mode restores the original input value after the backward pass, allowing it's use after other in-place modules, like [AddConstant](#nn.AddConstant).
