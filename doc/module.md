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
