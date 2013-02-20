local WeightedMSECriterion, parent = torch.class('nn.WeightedMSECriterion','nn.MSECriterion')

function WeightedMSECriterion:__init(w)
   parent.__init(self)
   self.weight = w:clone()
   self.buffer = torch.Tensor()
end

function WeightedMSECriterion:updateOutput(input,target)
	self.buffer:resizeAs(input):copy(target)
	if input:dim() - 1 == self.weight:dim() then
		for i=1,input:size(1) do
			self.buffer[i]:cmul(self.weight)
		end
	else
		self.buffer:cmul(self.weight)
	end
	return input.nn.MSECriterion_updateOutput(self, input, self.buffer)
end

function WeightedMSECriterion:updateGradInput(input, target)
   self.buffer:resizeAs(input):copy(target)
	if input:dim() - 1 == self.weight:dim() then
		for i=1,input:size(1) do
			self.buffer[i]:cmul(self.weight)
		end
	else
		self.buffer:cmul(self.weight)
	end
   return input.nn.MSECriterion_updateGradInput(self, input, self.buffer)
end
