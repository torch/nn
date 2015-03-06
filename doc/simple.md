<a name="nn.simplelayers.dok"/>
# Simple layers #
Simple Modules are used for various tasks like adapting Tensor methods 
and providing affine transformations :
 * Parameterized Modules :
   * [Linear](#nn.Linear) : a linear transformation ;
   * [SparseLinear](#nn.SparseLinear) : a linear transformation with sparse inputs ;
   * [Add](#nn.Add) : adds a bias term to the incoming data ;
   * [Mul](#nn.Mul) : multiply a single scalar factor to the incoming data ;
   * [CMul](#nn.CMul) : a component-wise multiplication to the incoming data ;
   * [CDiv](#nn.CDiv) : a component-wise division to the incoming data ;
   * [Euclidean](#nn.Euclidean) : the euclidean distance of the input to `k` mean centers ;
   * [WeightedEuclidean](#nn.WeightedEuclidean) : similar to [Euclidean](#nn.Euclidean), but additionally learns a diagonal covariance matrix ;
 * Modules that adapt basic Tensor methods :
   * [Copy](#nn.Copy) : a [copy](https://github.com/torch/torch7/blob/master/doc/tensor.md#torch.Tensor.copy) of the input with [type](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-or-string-typetype) casting ;
   * [Narrow](#nn.Narrow) : a [narrow](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-narrowdim-index-size) operation over a given dimension ;
   * [Replicate](#nn.Replicate) : [repeats](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-repeattensorresult-sizes) input `n` times along its first dimension ;
   * [Reshape](#nn.Reshape) : a [reshape](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchreshaperes-x-m-n) of the inputs ;
   * [View](#nn.View) : a [view](https://github.com/torch/torch7/blob/master/doc/tensor.md#result-viewresult-tensor-sizes) of the inputs ;
   * [Select](#nn.Select) : a [select](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-selectdim-index) over a given dimension ;
 * Modules that adapt mathematical Tensor methods :
   * [Max](#nn.Max) : a [max](https://github.com/torch/torch7/blob/master/doc/maths.md#torch.max) operation over a given dimension ;
   * [Min](#nn.Min) : a [min](https://github.com/torch/torch7/blob/master/doc/maths.md#torchminresval-resind-x) operation over a given dimension ;
   * [Mean](#nn.Mean) : a [mean](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchmeanres-x-dim) operation over a given dimension ;
   * [Sum](#nn.Sum) : a [sum](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchsumres-x) operation over a given dimension ;
   * [Exp](#nn.Exp) : an element-wise [exp](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchexpres-x) operation ;
   * [Abs](#nn.Abs) : an element-wise [abs](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchabsres-x) operation ;
   * [Power](#nn.Power) : an element-wise [pow](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchpowres-x) operation ;
   * [Square](#nn.Square) : an element-wise square operation ;
   * [Sqrt](#nn.Sqrt) : an element-wise [sqrt](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchsqrtres-x) operation ;
   * [MM](#nn.MM) : matrix-matrix multiplication (also supports batches of matrices) ;
 * Miscellaneous Modules :
   * [Identity](#nn.Identity) : forward input as-is to output (useful with [ParallelTable](table.md#nn.ParallelTable));
   * [Dropout](#nn.Dropout) : masks parts of the `input` using binary samples from a [bernoulli](http://en.wikipedia.org/wiki/Bernoulli_distribution) distribution ;
   

<a name="nn.Linear"/>
## Linear ##

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
## SparseLinear ##

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
indices should not exceed the stated dimensions of the input to the 
layer (10000 in the example).


<a name="nn.Dropout"/>
## Dropout ##

`module` = `nn.Dropout(p)`

During training, `Dropout` masks parts of the `input` using binary samples from 
a [bernoulli](http://en.wikipedia.org/wiki/Bernoulli_distribution) distribution.
Each `input` element has a probability of `p` of being dropped, i.e having its
commensurate output element be zero. This has proven an effective technique for 
regularization and preventing the co-adaptation of neurons 
(see [Hinton et al. 2012](http://arxiv.org/abs/1207.0580)). 

Furthermore, the ouputs are scaled by a factor of `1/(1-p)` during training. This allows the 
`input` to be simply forwarded as-is during evaluation.

In this example, we demonstrate how the call to [forward](module.md#output-forwardinput) samples 
different `outputs` to dropout (the zeros) given the same `input`:
```lua
module = nn.Dropout()

> x=torch.Tensor{{1,2,3,4},{5,6,7,8}}

> =module:forward(x)
  2   0   0   8
 10   0  14   0
[torch.DoubleTensor of dimension 2x4]

> =module:forward(x)
  0   0   6   0
 10   0   0   0
[torch.DoubleTensor of dimension 2x4]

```

[Backward](module.md#gradinput-backwardinput-gradoutput) drops out the gradients at the same location:
```lua
> =module:forward(x)
  0   4   0   0
 10  12   0  16
[torch.DoubleTensor of dimension 2x4]

> =module:backward(x,x:clone():fill(1))
 0  2  0  0
 2  2  0  2
[torch.DoubleTensor of dimension 2x4]

```
In both cases the `gradOutput` and `input` are scaled by `1/(1-p)`, which in this case is `2`.

During [evaluation](module.md#evaluate), `Dropout` does nothing more than 
forward the input such that all elements of the input are considered.
```lua
> module:evaluate()

> module:forward(x)
 1  2  3  4
 5  6  7  8
[torch.DoubleTensor of dimension 2x4]

```

We can return to training our model by first calling [Module:training()](module.md#training):
```lua
> module:training()

> return module:forward(x)
  2   4   6   0
  0   0   0  16
[torch.DoubleTensor of dimension 2x4]

```

When used, `Dropout` should normally be applied to the input of parameterized 
[Modules](module.md#nn.Module) like [Linear](#nn.Linear) 
or [SpatialConvolution](convolution.md#nn.SpatialConvolution).
A `p` of `0.5` (the default) is usually okay for hidden layers.
`Dropout` can sometimes be used successfully on the dataset inputs with a `p` around `0.2`.
It sometimes works best following [Transfer](transfer.md) Modules 
like [ReLU](transfer.md#nn.ReLU). All this depends a great deal on the dataset so its up 
to the user to try different combinations.


<a name="nn.Abs"/>
## Abs ##

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

![](image/abs.png)

<a name='nn.Add'/>
## Add ##

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
## Mul ##

`module` = `Mul()`

Applies a _single_ scaling factor to the incoming data, i.e.
_y= w x_, where _w_ is a scalar. 

Example:
```lua
y=torch.Tensor(5);  
mlp=nn.Sequential()
mlp:add(nn.Mul())

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

<a name='nn.CMul'/>
## CMul ##

`module` = `CMul(size)`

Applies a component-wise multiplication to the incoming data, i.e.
`y_i = w_i * x_i`. Argument `size` can be one or many numbers (sizes)
or a `torch.LongStorage`. For example, `nn.CMul(3,4,5)` is equivalent to 
`nn.CMul(torch.LongStorage{3,4,5})`.

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
## Max ##

`module` = `Max(dimension)`

Applies a max operation over dimension `dimension`.
Hence, if an `nxpxq` Tensor was given as input, and `dimension` = `2`
then an `nxq` matrix would be output.


<a name="nn.Min"/>
## Min ##

`module` = `Min(dimension)`

Applies a min operation over dimension `dimension`.
Hence, if an `nxpxq` Tensor was given as input, and `dimension` = `2`
then an `nxq` matrix would be output.


<a name="nn.Mean"/>
## Mean ##

`module` = `Mean(dimension)`

Applies a mean operation over dimension `dimension`.
Hence, if an `nxpxq` Tensor was given as input, and `dimension` = `2`
then an `nxq` matrix would be output.

<a name="nn.Sum"/>
## Sum ##

`module` = `Sum(dimension)`

Applies a sum operation over dimension `dimension`.
Hence, if an `nxpxq` Tensor was given as input, and `dimension` = `2`
then an `nxq` matrix would be output.


<a name="nn.Euclidean"/>
## Euclidean ##

`module` = `Euclidean(inputSize,outputSize)`

Outputs the Euclidean distance of the input to `outputSize` centers,
i.e. this layer has the weights `w_j`,  for `j` = `1`,..,`outputSize`, where
`w_j` are vectors of dimension `inputSize`. 

The distance `y_j` between center `j` and input `x` is formulated as
`y_j = || w_j - x ||`.

<a name="nn.WeightedEuclidean"/>
## WeightedEuclidean ##

`module` = `WeightedEuclidean(inputSize,outputSize)`

This module is similar to [Euclidean](#nn.Euclidean), but
additionally learns a separate diagonal covariance matrix across the
features of the input space _for each center_. 

In other words, for each of the `outputSize` centers `w_j`, there is 
a diagonal covariance matrices `c_j`, for `j` = `1`,..,`outputSize`, 
where `c_j` are stored as vectors of size `inputSize`.

The distance `y_j` between center `j` and input `x` is formulated as
`y_j = || c_j * (w_j - x) ||`.

<a name="nn.Identity"/>
## Identity ##

`module` = `Identity()`

Creates a module that returns whatever is input to it as output. 
This is useful when combined with the module 
[ParallelTable](table.md#nn.ParallelTable)
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

<a name="nn.Copy"/>
## Copy ##

`module` = `Copy(inputType,outputType,[forceCopy,dontCast])`

This layer copies the input to output with type casting from input
type from `inputType` to `outputType`. Unless `forceCopy` is true, when
the first two arguments are the same, the input isn't copied, only transfered
as the output. The default `forceCopy` is false. 
When `dontCast` is true, a call to `nn.Copy:type(type)` will not cast
the module's `output` and `gradInput` Tensors to the new type. The default 
is false.

<a name="nn.Narrow"/>
## Narrow ##

`module` = `Narrow(dimension, offset, length)`

Narrow is application of
[narrow](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-narrowdim-index-size) operation in a
module.

<a name="nn.Replicate"/>
## Replicate ##

`module` = `Replicate(nFeature)`

This class creates an output where the input is replicated
`nFeature` times along its first dimension. There is no memory
allocation or memory copy in this module. It sets the
[stride](https://github.com/torch/torch7/blob/master/doc/tensor.md#torch.Tensor.stride) along the first
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
## Reshape ##

`module` = `Reshape(dimension1, dimension2, ... [, batchMode])`

Reshapes an `nxpxqx..`  Tensor into a `dimension1xdimension2x...` Tensor,
taking the elements column-wise. 

The optional last argument `batchMode`, 
when `true` forces the first dimension of the input to be considered 
the batch dimension, and thus keep its size fixed. This is necessary when 
dealing with batch sizes of one. When `false`, it forces the 
entire input (including the first dimension) to be reshaped to the 
input size. Default `batchMode=nil`, which means that the module 
considers inputs with more elements than the produce of provided sizes, 
i.e. `dimension1xdimension2x...`, to be batches. 

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

  1   2   3   4   5   6   7   8
  9  10  11  12  13  14  15  16
[torch.Tensor of dimension 2x8]

> print(nn.Reshape(8,2):forward(x))

  1   2
  3   4
  5   6
  7   8
  9  10
 11  12
 13  14
 15  16
[torch.Tensor of dimension 8x2]

> print(nn.Reshape(16):forward(x))

  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
[torch.Tensor of dimension 16]
```

<a name="nn.View"/>
## View ##

`module` = `View(sizes)`

This module creates a new view of the input tensor using the `sizes` passed to
the constructor. The parameter `sizes` can either be a `LongStorage` or numbers.

The method `setNumInputDims()` allows to specify the expected number of dimensions
of the inputs of the modules. This makes it possible to use minibatch inputs when
using a size -1 for one of the dimensions.

Example 1:
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

> print(nn.View(2,8):forward(x))

  1   2   3   4   5   6   7   8
  9  10  11  12  13  14  15  16
[torch.DoubleTensor of dimension 2x8]

> print(nn.View(torch.LongStorage{8,2}):forward(x))

  1   2
  3   4
  5   6
  7   8
  9  10
 11  12
 13  14
 15  16
[torch.DoubleTensor of dimension 8x2]

> print(nn.View(16):forward(x))

  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
[torch.DoubleTensor of dimension 16]
```

Example 2:
```lua
> input = torch.Tensor(2,3)
> minibatch = torch.Tensor(5,2,3)
> m = nn.View(-1):setNumInputDims(2)
> print(#m:forward(input))

 6
[torch.LongStorage of size 2]

> print(#m:forward(minibatch))

 5
 6
[torch.LongStorage of size 2]

```

<a name="nn.Select"/>
## Select ##

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

This can be used in conjunction with [Concat](containers.md#nn.Concat)
to emulate the behavior 
of [Parallel](containers.md#nn.Parallel), or to select various parts of an input Tensor to 
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
## Exp ##

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
![](image/exp.png)


<a name="nn.Square"/>
## Square ##

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
![](image/square.png)

<a name="nn.Sqrt"/>
## Sqrt ##

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
![](image/sqrt.png)

<a name="nn.Power"/>
## Power ##

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
![](image/power.png)

<a name="nn.MM"/>
## MM ##

`module` = `nn.MM(transA, transB)`

Performs multiplications on one or more pairs of matrices.
If `transA` is set, the first matrix is transposed before multiplication.
If `transB` is set, the second matrix is transposed before multiplication.
By default, the matrices do not get transposed.

The module also accepts 3D inputs which are interpreted as batches of matrices.
When using batches, the first input matrix should be of size `b x m x n` and the
second input matrix should be of size `b x n x p` (assuming `transA` and `transB`
are not set).

```lua
model = nn.MM()
A = torch.randn(b, m, n)
B = torch.randn(b, n, p)
C = model.forward({A, B})  -- C will be of size `b x m x n`
```
