local ELU, parent = torch.class('nn.ELU', 'nn.Module')

--[[
   Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter
   Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
   http://arxiv.org/pdf/1511.07289.pdf
--]]

function ELU:__init(alpha)
   parent.__init(self)
   self.alpha = alpha or 1
   assert(type(self.alpha) == 'number')
end

function ELU:updateOutput(input)
   input.THNN.ELU_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.alpha
   )
   return self.output
end

function ELU:updateGradInput(input, gradOutput)
  input.THNN.ELU_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata(),
      self.alpha
   )
   return self.gradInput
end

function ELU:__tostring__()
  return string.format('%s (alpha:%f)', torch.type(self), self.alpha)
end
