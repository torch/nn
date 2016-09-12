[![Build Status](https://travis-ci.org/torch/nn.svg?branch=master)](https://travis-ci.org/torch/nn)
<a name="nn.dok"></a>
# Neural Network Package #

This package provides an easy and modular way to build and train simple or complex neural networks using [Torch](https://github.com/torch/torch7/blob/master/README.md):
 * Modules are the bricks used to build neural networks. Each are themselves neural networks, but can be combined with other networks using containers to create complex neural networks:
   * [Module](nn/blob/master/doc/module.md#nn.Module): abstract class inherited by all modules;
   * [Containers](nn/blob/master/doc/containers.md#nn.Containers): container classes like [`Sequential`](nn/blob/master/doc/containers.md#nn.Sequential), [`Parallel`](nn/blob/master/doc/containers.md#nn.Parallel) and [`Concat`](nn/blob/master/doc/containers.md#nn.Concat);
   * [Transfer functions](nn/blob/master/doc/transfer.md#nn.transfer.dok): non-linear functions like [`Tanh`](nn/blob/master/doc/transfer.md#nn.Tanh) and [`Sigmoid`](nn/blob/master/doc/transfer.md#nn.Sigmoid);
   * [Simple layers](nn/blob/master/doc/simple.md#nn.simplelayers.dok): like [`Linear`](nn/blob/master/doc/simple.md#nn.Linear), [`Mean`](nn/blob/master/doc/simple.md#nn.Mean), [`Max`](nn/blob/master/doc/simple.md#nn.Max) and [`Reshape`](nn/blob/master/doc/simple.md#nn.Reshape);
   * [Table layers](nn/blob/master/doc/table.md#nn.TableLayers): layers for manipulating `table`s like [`SplitTable`](nn/blob/master/doc/table.md#nn.SplitTable), [`ConcatTable`](nn/blob/master/doc/table.md#nn.ConcatTable) and [`JoinTable`](nn/blob/master/doc/table.md#nn.JoinTable);
   * [Convolution layers](nn/blob/master/doc/convolution.md#nn.convlayers.dok): [`Temporal`](nn/blob/master/doc/convolution.md#nn.TemporalModules),  [`Spatial`](nn/blob/master/doc/convolution.md#nn.SpatialModules) and [`Volumetric`](nn/blob/master/doc/convolution.md#nn.VolumetricModules) convolutions;
 * Criterions compute a gradient according to a given loss function given an input and a target:
   * [Criterions](nn/blob/master/doc/criterion.md#nn.Criterions): a list of all criterions, including [`Criterion`](nn/blob/master/doc/criterion.md#nn.Criterion), the abstract class;
   * [`MSECriterion`](nn/blob/master/doc/criterion.md#nn.MSECriterion): the Mean Squared Error criterion used for regression;
   * [`ClassNLLCriterion`](nn/blob/master/doc/criterion.md#nn.ClassNLLCriterion): the Negative Log Likelihood criterion used for classification;
 * Additional documentation:
   * [Overview](nn/blob/master/doc/overview.md#nn.overview.dok) of the package essentials including modules, containers and training;
   * [Training](nn/blob/master/doc/training.md#nn.traningneuralnet.dok): how to train a neural network using [optim](https://github.com/torch/optim);
   * [Testing](nn/blob/master/doc/testing.md): how to test your modules.
   * [Experimental Modules](https://github.com/clementfarabet/lua---nnx/blob/master/README.md): a package containing experimental modules and criteria.
