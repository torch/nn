local Identity, parent = torch.class('nn.Identity', 'nn.Module')

function Identity:__init()
   parent.__init(self)
   self.tensorOutput = torch.Tensor{}
   self.output = self.tensorOutput
   self.tensorGradInput = torch.Tensor{}
   self.gradInput = self.tensorGradInput
end

function Identity:updateOutput(input)
   if torch.isTensor(input) then
      self.tensorOutput:set(input)
      self.output = self.tensorOutput
   else
      self.output = input
   end
   return self.output
end


function Identity:updateGradInput(input, gradOutput)
   if torch.isTensor(gradOutput) then
      self.tensorGradInput:set(gradOutput)
      self.gradInput = self.tensorGradInput
   else
      self.gradInput = gradOutput
   end
   return self.gradInput
end

function Identity:clearState()
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
