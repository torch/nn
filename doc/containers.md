<a name="nn.Containers"></a>
# Containers #
Complex neural networks are easily built using container classes:

  * [Container](#nn.Container) : abstract class inherited by containers ;
    * [Sequential](#nn.Sequential) : plugs layers in a feed-forward fully connected manner ;
    * [Parallel](#nn.Parallel) : applies its `ith` child module to the  `ith` slice of the input Tensor ;
    * [Concat](#nn.Concat) : concatenates in one layer several modules along dimension `dim` ;
      * [DepthConcat](#nn.DepthConcat) : like Concat, but adds zero-padding when non-`dim` sizes don't match;
 
See also the [Table Containers](#nn.TableContainers) for manipulating tables of [Tensors](https://github.com/torch/torch7/blob/master/doc/tensor.md).

<a name="nn.Container"></a>
## Container ##

This is an abstract [Module](module.md#nn.Module) class which declares methods defined in all containers.
It reimplements many of the Module methods such that calls are propagated to the 
contained modules. For example, a call to [zeroGradParameters](module.md#nn.Module.zeroGradParameters)
will be propagated to all contained modules.

<a name="nn.Container.add"></a>
### add(module) ###
Adds the given `module` to the container. The order is important

<a name="nn.Container.get"></a>
### get(index) ###
Returns the contained modules at index `index`.

<a name="nn.Container.size"></a>
### size() ###
Returns the number of contained modules.

<a name="nn.Sequential"></a>
## Sequential ##

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

<a name="nn.Sequential.remove"></a>
### remove([index]) ###

Remove the module at the given `index`. If `index` is not specified, remove the last layer.

```lua
model = nn.Sequential()
model:add(nn.Linear(10, 20))
model:add(nn.Linear(20, 20))
model:add(nn.Linear(20, 30))
model:remove(2)
> model
nn.Sequential {
  [input -> (1) -> (2) -> output]
  (1): nn.Linear(10 -> 20)
  (2): nn.Linear(20 -> 30)
}
```


<a name="nn.Sequential.insert"></a>
### insert(module, [index]) ###

Inserts the given `module` at the given `index`. If `index` is not specified, the incremented length of the sequence is used and so this is equivalent to use `add(module)`.

```lua
model = nn.Sequential()
model:add(nn.Linear(10, 20))
model:add(nn.Linear(20, 30))
model:insert(nn.Linear(20, 20), 2)
> model
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> output]
  (1): nn.Linear(10 -> 20)
  (2): nn.Linear(20 -> 20)      -- The inserted layer
  (3): nn.Linear(20 -> 30)
}
```



<a name="nn.Parallel"></a>
## Parallel ##

`module` = `Parallel(inputDimension,outputDimension)`

Creates a container module that applies its `ith` child module to the  `ith` slice of the input Tensor by using [select](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-selectdim-index) 
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


<a name="nn.Concat"></a>
## Concat ##

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

<a name="nn.DepthConcat"></a>
## DepthConcat ##

```lua
module = nn.DepthConcat(dim)
```
DepthConcat concatenates the output of one layer of "parallel" modules along the
provided dimension `dim`: they take the same inputs, and their output is
concatenated. For dimensions other than `dim` having different sizes,
the smaller tensors are copied in the center of the output tensor, 
effectively padding the borders with zeros.

The module is particularly useful for concatenating the output of [Convolutions](convolution.md) 
along the depth dimension (i.e. `nOutputFrame`). 
This is used to implement the *DepthConcat* layer 
of the [Going deeper with convolutions](http://arxiv.org/pdf/1409.4842v1.pdf) article.
The normal [Concat](#nn.Concat) Module can't be used since the spatial 
dimensions (height and width) of the output Tensors requiring concatenation 
may have different values. To deal with this, the output uses the largest 
spatial dimensions and adds zero-padding around the smaller Tensors.
```lua
inputSize = 3
outputSize = 2
input = torch.randn(inputSize,7,7)
mlp=nn.DepthConcat(1);
mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 1, 1))
mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 3, 3))
mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 4, 4))
print(mlp:forward(input))
```
which gives the output:
```lua
(1,.,.) = 
 -0.2874  0.6255  1.1122  0.4768  0.9863 -0.2201 -0.1516
  0.2779  0.9295  1.1944  0.4457  1.1470  0.9693  0.1654
 -0.5769 -0.4730  0.3283  0.6729  1.3574 -0.6610  0.0265
  0.3767  1.0300  1.6927  0.4422  0.5837  1.5277  1.1686
  0.8843 -0.7698  0.0539 -0.3547  0.6904 -0.6842  0.2653
  0.4147  0.5062  0.6251  0.4374  0.3252  0.3478  0.0046
  0.7845 -0.0902  0.3499  0.0342  1.0706 -0.0605  0.5525

(2,.,.) = 
 -0.7351 -0.9327 -0.3092 -1.3395 -0.4596 -0.6377 -0.5097
 -0.2406 -0.2617 -0.3400 -0.4339 -0.3648  0.1539 -0.2961
 -0.7124 -1.2228 -0.2632  0.1690  0.4836 -0.9469 -0.7003
 -0.0221  0.1067  0.6975 -0.4221 -0.3121  0.4822  0.6617
  0.2043 -0.9928 -0.9500 -1.6107  0.1409 -1.3548 -0.5212
 -0.3086 -0.0298 -0.2031  0.1026 -0.5785 -0.3275 -0.1630
  0.0596 -0.6097  0.1443 -0.8603 -0.2774 -0.4506 -0.5367

(3,.,.) = 
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000 -0.7326  0.3544  0.1821  0.4796  1.0164  0.0000
  0.0000 -0.9195 -0.0567 -0.1947  0.0169  0.1924  0.0000
  0.0000  0.2596  0.6766  0.0939  0.5677  0.6359  0.0000
  0.0000 -0.2981 -1.2165 -0.0224 -1.1001  0.0008  0.0000
  0.0000 -0.1911  0.2912  0.5092  0.2955  0.7171  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000

(4,.,.) = 
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000 -0.8263  0.3646  0.6750  0.2062  0.2785  0.0000
  0.0000 -0.7572  0.0432 -0.0821  0.4871  1.9506  0.0000
  0.0000 -0.4609  0.4362  0.5091  0.8901 -0.6954  0.0000
  0.0000  0.6049 -0.1501 -0.4602 -0.6514  0.5439  0.0000
  0.0000  0.2570  0.4694 -0.1262  0.5602  0.0821  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000

(5,.,.) = 
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000  0.3158  0.4389 -0.0485 -0.2179  0.0000  0.0000
  0.0000  0.1966  0.6185 -0.9563 -0.3365  0.0000  0.0000
  0.0000 -0.2892 -0.9266 -0.0172 -0.3122  0.0000  0.0000
  0.0000 -0.6269  0.5349 -0.2520 -0.2187  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000

(6,.,.) = 
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000  1.1148  0.2324 -0.1093  0.5024  0.0000  0.0000
  0.0000 -0.2624 -0.5863  0.3444  0.3506  0.0000  0.0000
  0.0000  0.1486  0.8413  0.6229 -0.0130  0.0000  0.0000
  0.0000  0.8446  0.3801 -0.2611  0.8140  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
[torch.DoubleTensor of dimension 6x7x7]
```
Note how the last 2 of 6 filter maps have 1 column of zero-padding 
on the left and top, as well as 2 on the right and bottom. 
This is inevitable when the component
module output tensors non-`dim` sizes aren't all odd or even. 
Such that in order to keep the mappings aligned, one need 
only ensure that these be all odd (or even).

<a name="nn.TableContainers"></a>
## Table Containers ##
While the above containers are used for manipulating input [Tensors](https://github.com/torch/torch7/blob/master/doc/tensor.md), table containers are used for manipulating tables :
 * [ConcatTable](table.md#nn.ConcatTable)
 * [ParallelTable](table.md#nn.ParallelTable)

These, along with all other modules for manipulating tables can be found [here](table.md).
