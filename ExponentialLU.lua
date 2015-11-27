local ExponentialLU, parent = torch.class('nn.ExponentialLU','nn.Module')

--[[
Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter
Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
http://arxiv.org/pdf/1511.07289v1.pdf
--]]

function ExponentialLU:__init(alpha)
   parent.__init(self)
   self.alpha = alpha or 1
   if (alpha and type(alpha) ~= 'number') then
	   error('nn.ExponentialLU(alpha)')
   end
end

function ExponentialLU:updateOutput(input)
   self.output = input:clone()
   self.output[input:lt(0)] = (self.output[input:lt(0)]:exp()-1) * self.alpha
   return self.output
end

function ExponentialLU:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput:clone()
   self.gradInput[input:lt(0)] = self.gradInput[input:lt(0)]:cmul( self.output[input:lt(0)] + self.alpha )
   return self.gradInput
end
