local BC, parent = torch.class('nn.SpatialBatchCentering', 'nn.BatchCentering')

-- expected dimension of input
BC.nDim = 4
