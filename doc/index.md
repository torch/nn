[![Build Status](https://travis-ci.org/torch/nn.svg?branch=master)](https://travis-ci.org/torch/nn)
<a name="nn.dok"></a>
# Neural Network Package #

This package provides an easy and modular way to build and train simple or complex neural networks using [Torch](https://github.com/torch/torch7/blob/master/README.md):
  
  * Modules are the bricks used to build neural networks. Each are themselves neural networks, but can be combined with other networks using containers to create complex neural networks:
    * [Module](module.md#nn.Module) : abstract class inherited by all modules;
    * [Containers](containers.md#nn.Containers) : container classes like [Sequential](containers.md#nn.Sequential), [Parallel](containers.md#nn.Parallel) and [Concat](containers.md#nn.Concat);
    * [Transfer functions](transfer.md#nn.transfer.dok) : non-linear functions like [Tanh](transfer.md#nn.Tanh) and [Sigmoid](transfer.md#nn.Sigmoid);
    * [Simple layers](simple.md#nn.simplelayers.dok) : like [Linear](simple.md#nn.Linear), [Mean](simple.md#nn.Mean), [Max](simple.md#nn.Max) and [Reshape](simple.md#nn.Reshape); 
    * [Table layers](table.md#nn.TableLayers) : layers for manipulating tables like [SplitTable](table.md#nn.SplitTable), [ConcatTable](table.md#nn.ConcatTable) and [JoinTable](table.md#nn.JoinTable);
    * [Convolution layers](convolution.md#nn.convlayers.dok) : [Temporal](convolution.md#nn.TemporalModules),  [Spatial](convolution.md#nn.SpatialModules) and [Volumetric](convolution.md#nn.VolumetricModules) convolutions ; 
  * Criterions compute a gradient according to a given loss function given an input and a target:
    * [Criterions](criterion.md#nn.Criterions) : a list of all criterions, including [Criterion](criterion.md#nn.Criterion), the abstract class;
    * [MSECriterion](criterion.md#nn.MSECriterion) : the Mean Squared Error criterion used for regression; 
    * [ClassNLLCriterion](criterion.md#nn.ClassNLLCriterion) : the Negative Log Likelihood criterion used for classification;
  * Additional documentation :
    * [Overview](overview.md#nn.overview.dok) of the package essentials including modules, containers and training;
    * [Training](training.md#nn.traningneuralnet.dok) : how to train a neural network using [StochasticGradient](training.md#nn.StochasticGradient);
    * [Testing](testing.md) : how to test your modules.
    * [Experimental Modules](https://github.com/clementfarabet/lua---nnx/blob/master/README.md) : a package containing experimental modules and criteria.

