<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
# Table of Content

- [Testing](#testing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Testing ##
For those who want to implement their own modules, we suggest using
the `nn.Jacobian` class for testing the derivatives of their class,
together with the [torch.Tester](https://github.com/torch/torch7/blob/master/doc/tester.md) class. The sources
of `nn` package contains sufficiently many examples of such tests.
