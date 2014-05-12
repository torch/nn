<a name="nn.Containers"/>
# Containers #
Complex neural networks are easily built using container classes:
 * [Sequential](#nn.Sequential) : plugs layers in a feed-forward fully connected manner ;
 * [Parallel](#nn.Parallel) : applies its `ith` child module to the  `ith` slice of the input Tensor ;
 * [Concat](#nn.Concat) : concatenates in one layer several modules ;
 
See also the [Table Containers](#nn.TableContainers) for manipulating tables of [Tensors](https://github.com/torch/torch7/blob/master/doc/tensor.md).

<a name="nn.Sequential"/>
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

<a name="nn.Parallel"/>
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

<a name="nn.Concat"/>
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

<a name="nn.TableContainers"/>
## Table Containers ##
While the above containers are used for manipulating input [Tensors](https://github.com/torch/torch7/blob/master/doc/tensor.md), table containers are used for manipulating tables :
 * [ConcatTable](table.md#nn.ConcatTable>)
 * [ParallelTable](table.md#nn.ParallelTable)

These, along with all other modules for manipulating tables can be found [here](doc/table.md).
