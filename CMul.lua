local CMul, parent = torch.class('nn.CMul', 'nn.Module')

function CMul:__init(inputSize)
   parent.__init(self)
  
   self.weight = torch.Tensor(inputSize)
   self.gradWeight = torch.Tensor(inputSize)
   
   -- state
   -- self.gradInput:resize(inputSize)
   self.output:resize(inputSize) 

   self:reset()
end
 
function CMul:reset()
   self.weight:fill(1)
end

function CMul:updateOutput(input)
   self.output:copy(input);
   self.output:cmul(self.weight);
   return self.output
end

function CMul:updateGradInput(input, gradOutput)
   if self.gradInput then
      self.gradInput:resizeAs(input)
      self.gradInput:zero()
      self.gradInput:addcmul(1, self.weight, gradOutput)
      return self.gradInput
   end
end

function CMul:accGradParameters(input, gradOutput, scale)
   self.gradWeight:addcmul(scale or 1, input, gradOutput)
end
