require('torch')
require('libnn')

include('ErrorMessages.lua')
include('Module.lua')

include('Container.lua')
include('Concat.lua')
include('Parallel.lua')
include('Sequential.lua')
include('DepthConcat.lua')

include('Linear.lua')
include('SparseLinear.lua')
include('Reshape.lua')
include('View.lua')
include('Select.lua')
include('Narrow.lua')
include('Replicate.lua')
include('Transpose.lua')
include('BatchNormalization.lua')
include('ZeroGrad.lua')

include('Copy.lua')
include('Min.lua')
include('Max.lua')
include('Mean.lua')
include('Sum.lua')
include('CMul.lua')
include('Mul.lua')
include('MulConstant.lua')
include('Add.lua')
include('AddConstant.lua')
include('Dropout.lua')

include('CAddTable.lua')
include('CDivTable.lua')
include('CMulTable.lua')
include('CSubTable.lua')

include('Euclidean.lua')
include('WeightedEuclidean.lua')
include('PairwiseDistance.lua')
include('CosineDistance.lua')
include('DotProduct.lua')

include('Exp.lua')
include('Log.lua')
include('HardTanh.lua')
include('LogSigmoid.lua')
include('LogSoftMax.lua')
include('Sigmoid.lua')
include('SoftMax.lua')
include('SoftMin.lua')
include('SoftPlus.lua')
include('SoftSign.lua')
include('Tanh.lua')
include('TanhShrink.lua')
include('Abs.lua')
include('Power.lua')
include('Square.lua')
include('Sqrt.lua')
include('HardShrink.lua')
include('SoftShrink.lua')
include('Threshold.lua')
include('ReLU.lua')
include('PReLU.lua')

include('LookupTable.lua')
include('SpatialConvolution.lua')
include('SpatialFullConvolution.lua')
include('SpatialFullConvolutionMap.lua')
include('SpatialConvolutionMM.lua')
include('SpatialConvolutionMM_BHWD.lua')
include('SpatialConvolutionCUDA.lua')
include('SpatialConvolutionMap.lua')
include('SpatialSubSampling.lua')
include('SpatialMaxPooling.lua')
include('SpatialMaxPoolingCUDA.lua')
include('SpatialLPPooling.lua')
include('SpatialAveragePooling.lua')
include('SpatialAdaptiveMaxPooling.lua')
include('TemporalConvolution.lua')
include('TemporalSubSampling.lua')
include('TemporalMaxPooling.lua')
include('SpatialSubtractiveNormalization.lua')
include('SpatialDivisiveNormalization.lua')
include('SpatialContrastiveNormalization.lua')
include('SpatialZeroPadding.lua')
include('SpatialUpSamplingNearest.lua')
include('SpatialBatchNormalization.lua')

include('VolumetricConvolution.lua')
include('VolumetricMaxPooling.lua')

include('ParallelTable.lua')
include('ConcatTable.lua')
include('SplitTable.lua')
include('JoinTable.lua')
include('SelectTable.lua')
include('MixtureTable.lua')
include('CriterionTable.lua')
include('FlattenTable.lua')
include('Identity.lua')

include('Criterion.lua')
include('MSECriterion.lua')
include('MarginCriterion.lua')
include('AbsCriterion.lua')
include('ClassNLLCriterion.lua')
include('DistKLDivCriterion.lua')
include('MultiCriterion.lua')
include('L1HingeEmbeddingCriterion.lua')
include('HingeEmbeddingCriterion.lua')
include('CosineEmbeddingCriterion.lua')
include('MarginRankingCriterion.lua')
include('MultiMarginCriterion.lua')
include('MultiLabelMarginCriterion.lua')
include('L1Cost.lua')
include('L1Penalty.lua')
include('WeightedMSECriterion.lua')
include('BCECriterion.lua')
include('CrossEntropyCriterion.lua')

include('StochasticGradient.lua')

include('MM.lua')

include('Jacobian.lua')
include('SparseJacobian.lua')
include('hessian.lua')
include('test.lua')

return nn
