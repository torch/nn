local PReLU, parent = torch.class('nn.PReLU','nn.Module')

function PReLU:__init(nOutputPlane)
   parent.__init(self)
   -- if no argument provided, use shared model (weight is scalar)
   self.nOutputPlane = nOutputPlane or 0
   self.weight = torch.Tensor(nOutputPlane or 1):fill(0.25)
   self.gradWeight = torch.Tensor(nOutputPlane or 1)
   self.gradWeightBuf = torch.Tensor()
   self.gradWeightBuf2 = torch.Tensor()
end

function PReLU:updateOutput(input)
   input.nn.PReLU_updateOutput(self, input)
   return self.output
end

function PReLU:updateGradInput(input, gradOutput)
   input.nn.PReLU_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function PReLU:accGradParameters(input, gradOutput, scale)
   input.nn.PReLU_accGradParameters(self, input, gradOutput, scale)
   return self.gradWeight
end
