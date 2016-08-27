require('torch')

nn = {} -- define the global nn table

require('nn.THNN')

require('nn.utils')


require('nn.ErrorMessages')
require('nn.Module')

require('nn.Container')
require('nn.Concat')
require('nn.Parallel')
require('nn.Sequential')
require('nn.DepthConcat')
require('nn.Bottle')

require('nn.Linear')
require('nn.Bilinear')
require('nn.PartialLinear')
require('nn.SparseLinear')
require('nn.Reshape')
require('nn.View')
require('nn.Contiguous')
require('nn.Select')
require('nn.Narrow')
require('nn.Index')
require('nn.Squeeze')
require('nn.Unsqueeze')
require('nn.Replicate')
require('nn.Transpose')
require('nn.BatchNormalization')
require('nn.Padding')
require('nn.GradientReversal')
require('nn.MaskedSelect')

require('nn.Copy')
require('nn.Min')
require('nn.Max')
require('nn.Sum')
require('nn.Mean')
require('nn.CMul')
require('nn.Mul')
require('nn.MulConstant')
require('nn.Add')
require('nn.AddConstant')
require('nn.Dropout')
require('nn.SpatialDropout')
require('nn.VolumetricDropout')

require('nn.CAddTable')
require('nn.CDivTable')
require('nn.CMulTable')
require('nn.CSubTable')
require('nn.CMaxTable')
require('nn.CMinTable')

require('nn.Euclidean')
require('nn.WeightedEuclidean')
require('nn.PairwiseDistance')
require('nn.CosineDistance')
require('nn.DotProduct')
require('nn.Normalize')
require('nn.Cosine')

require('nn.Exp')
require('nn.Log')
require('nn.HardTanh')
require('nn.Clamp')
require('nn.LogSigmoid')
require('nn.LogSoftMax')
require('nn.Sigmoid')
require('nn.SoftMax')
require('nn.SoftMaxAtTest')
require('nn.SoftMin')
require('nn.SoftPlus')
require('nn.SoftSign')
require('nn.Tanh')
require('nn.TanhShrink')
require('nn.Abs')
require('nn.Power')
require('nn.Square')
require('nn.Sqrt')
require('nn.HardShrink')
require('nn.SoftShrink')
require('nn.Threshold')
require('nn.ReLU')
require('nn.ReLU6')
require('nn.PReLU')
require('nn.LeakyReLU')
require('nn.SpatialSoftMax')
require('nn.SpatialSoftMaxAtTest')
require('nn.RReLU')
require('nn.ELU')

require('nn.LookupTable')
require('nn.SpatialConvolution')
require('nn.SpatialConvolutionLocal')
require('nn.SpatialFullConvolution')
require('nn.SpatialFullConvolutionMap')
require('nn.SpatialConvolutionMM')
require('nn.SpatialConvolutionMap')
require('nn.SpatialDilatedConvolution')
require('nn.SpatialSubSampling')
require('nn.SpatialMaxPooling')
require('nn.SpatialDilatedMaxPooling')
require('nn.SpatialMaxUnpooling')
require('nn.SpatialFractionalMaxPooling')
require('nn.SpatialLPPooling')
require('nn.SpatialAveragePooling')
require('nn.SpatialAdaptiveMaxPooling')
require('nn.TemporalConvolution')
require('nn.TemporalSubSampling')
require('nn.TemporalMaxPooling')
require('nn.TemporalDynamicKMaxPooling')
require('nn.SpatialSubtractiveNormalization')
require('nn.SpatialDivisiveNormalization')
require('nn.SpatialContrastiveNormalization')
require('nn.SpatialCrossMapLRN')
require('nn.SpatialZeroPadding')
require('nn.SpatialReflectionPadding')
require('nn.SpatialReplicationPadding')
require('nn.SpatialUpSamplingNearest')
require('nn.SpatialUpSamplingBilinear')
require('nn.SpatialBatchNormalization')

require('nn.VolumetricConvolution')
require('nn.VolumetricFullConvolution')
require('nn.VolumetricDilatedConvolution')
require('nn.VolumetricMaxPooling')
require('nn.VolumetricDilatedMaxPooling')
require('nn.VolumetricMaxUnpooling')
require('nn.VolumetricAveragePooling')
require('nn.VolumetricBatchNormalization')
require('nn.VolumetricReplicationPadding')

require('nn.GPU')

require('nn.ParallelTable')
require('nn.Identity')
require('nn.ConcatTable')
require('nn.SplitTable')
require('nn.JoinTable')
require('nn.SelectTable')
require('nn.MixtureTable')
require('nn.CriterionTable')
require('nn.FlattenTable')
require('nn.NarrowTable')
require('nn.MapTable')

require('nn.Criterion')
require('nn.MSECriterion')
require('nn.SmoothL1Criterion')
require('nn.WeightedSmoothL1Criterion')
require('nn.MarginCriterion')
require('nn.SoftMarginCriterion')
require('nn.AbsCriterion')
require('nn.ClassNLLCriterion')
require('nn.WeightedClassNLLCriterion')
require('nn.SpatialClassNLLCriterion')
require('nn.SpatialWeightedClassNLLCriterion')
require('nn.ClassSimplexCriterion')
require('nn.DistKLDivCriterion')
require('nn.MultiCriterion')
require('nn.L1HingeEmbeddingCriterion')
require('nn.HingeEmbeddingCriterion')
require('nn.CosineEmbeddingCriterion')
require('nn.MarginRankingCriterion')
require('nn.MultiMarginCriterion')
require('nn.MultiLabelMarginCriterion')
require('nn.MultiLabelSoftMarginCriterion')
require('nn.L1Cost')
require('nn.L1Penalty')
require('nn.WeightedMSECriterion')
require('nn.BCECriterion')
require('nn.CrossEntropyCriterion')
require('nn.WeightedCrossEntropyCriterion')
require('nn.SpatialWeightedCrossEntropyCriterion')
require('nn.ParallelCriterion')

require('nn.StochasticGradient')

require('nn.MM')
require('nn.MV')

require('nn.Jacobian')
require('nn.SparseJacobian')
require('nn.hessian')
require('nn.test')

return nn
