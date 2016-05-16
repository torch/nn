local SpatialSoftMaxAtTest, parent = torch.class('nn.SpatialSoftMaxAtTest', 'nn.Module')

function SpatialSoftMaxAtTest:__init()
   parent.__init(self)

   self.train = true
   self.sfm = nn.SpatialSoftMax()
   self.id = nn.Identity()
end

function SpatialSoftMaxAtTest:updateOutput(input)
   if input:type() == 'torch.CudaTensor' then
     self.sfm:cuda()
     self.id:cuda()
   end

   if self.train then
      self.id:updateOutput(input)
      self.output = self.id.output
   else
      self.sfm:updateOutput(input)
      self.output = self.sfm.output
   end

   return self.output
end

function SpatialSoftMaxAtTest:updateGradInput(input, gradOutput)
   if input:type() == 'torch.CudaTensor' then
     self.sfm:cuda()
     self.id:cuda()
   end

   if self.train then
      self.id:updateGradInput(input, gradOutput)
      self.gradInput = self.id.gradInput
      return self.gradInput
   else
      error('backprop only defined while training')
   end
end

function SpatialSoftMaxAtTest:clearState()
   -- don't call set because it might reset referenced tensors
   local function clear(f)
      if self[f] then
         if torch.isTensor(self[f]) then
            self[f] = self[f].new()
         elseif type(self[f]) == 'table' then
            self[f] = {}
         else
            self[f] = nil
         end
      end
   end
   clear('output')
   clear('gradInput')
   return self
end
