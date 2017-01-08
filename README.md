[![Build Status](https://travis-ci.org/torch/nn.svg?branch=master)](https://travis-ci.org/torch/nn)
<a name="nn.dok"></a>
# Neural Network Package #

**This branch of 'nn' package adds support for 'SpatialDepthWiseConvolution', which is similar to TensorFlow's '[depthwise_conv2d](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard4/tf.nn.depthwise_conv2d.md)'.**

CUDA implementation for this nn package: [cunn](https://github.com/stooloveu/cunn/)

## Installing from source
```bash
git clone https://github.com/stooloveu/nn
cd nn
luarocks make rocks/cunn-scm-1.rockspec
```

## To use 'SpatialDepthWiseConvolution'

```lua
module = nn.SpatialDepthWiseConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH])
```
Applies a 2D depth-wise convolution over an input image composed of several input planes. The `input` tensor in
`forward(input)` is expected to be a 3D tensor (`nInputPlane x height x width`).

It is similar to 'SpatialConvolution', but its output has layers for each input channel. The most noticiable difference is the output dimension of 'SpatialConvolution' is \[nOutputPlane\]\[outputWidth\]\[outputHeight\], while for 'SpatialDepthWiseConvolution' it is  \[nInputPlane\]\[nOutputPlane\]\[outputWidth\]\[outputHeight\]

The parameters are the following:
  * `nInputPlane`: The number of expected input planes in the image given into `forward()`.
  * `nOutputPlane`: The number of output planes the convolution layer will produce.
  * `kW`: The kernel width of the convolution
  * `kH`: The kernel height of the convolution
  * `dW`: The step of the convolution in the width dimension. Default is `1`.
  * `dH`: The step of the convolution in the height dimension. Default is `1`.
  * `padW`: Additional zeros added to the input plane data on both sides of width axis. Default is `0`. `(kW-1)/2` is often used here.
  * `padH`: Additional zeros added to the input plane data on both sides of height axis. Default is `0`. `(kH-1)/2` is often used here.

Note that depending of the size of your kernel, several (of the last)
columns or rows of the input image might be lost. It is up to the user to
add proper padding in images.

If the input image is a 3D tensor `nInputPlane x height x width`, the output image size
will be `nOutputPlane x nInputPlane x oheight x owidth` where
```lua
owidth  = floor((width  + 2*padW - kW) / dW + 1)
oheight = floor((height + 2*padH - kH) / dH + 1)
```

The parameters of the convolution can be found in `self.weight` (Tensor of
size `nOutputPlane x nInputPlane x kH x kW`) and `self.bias` (Tensor of
size `nOutputPlane x nInputPlane`). The corresponding gradients can be found in
`self.gradWeight` and `self.gradBias`.

The output value of the layer can be described as:
```
output[i][j] = input[j] * weight[i][j] + b[i][j], i = 1, ..., nOutputPlane, j = 1, ..., nInputPlane
```

## Example

```lua
require 'cutorch'
require 'nn'

local nip = 2;
local nop = 3;
local kW = 3;
local kH = 3;
local iW = 9;
local iH = 9;

local model = nn.Sequential()
model:add(nn.SpatialDepthWiseConvolution(nip, nop, kW, kW))

local weight = torch.rand(nop, nip, kW, kW)
model:get(1).weight = weight

local bias = torch.rand(nop, nip)
model:get(1).bias = bias

local gradOutput = torch.rand(nop, nip, iW - kW + 1, iH - kH + 1)

local output = model:forward(input)
local gradInput = model:backward((input, gradOutput)
```

## To use 'nn' package

See <https://github.com/torch/nn/blob/master/README.md>
