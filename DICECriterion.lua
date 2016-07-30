--[[
	A  statistic used for comparing the similarity of two samples
	Sorensen's original formula =  2 * |X n Y|
	                              ------------
	                                |X| + |Y|

	Note: |X| and |Y| are the numbers of species in the two samples. 
	The result is the quotient of similarity and ranges between 0 and 1.

	Ref 1: https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient
	Ref 2: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19

	Author: Olalekan Ogunmolu, July 2016
			patlekano@gmail.com
]]

local DICECriterion, parent = torch.class('nn.DICECriterion', 'nn.Criterion')

local eps = 1e-12

function DICECriterion:_init(weights)
	parent._init(self)

	if weights then
	   assert(weights:dim() == 1, "weights input should be 1-D Tensor")
	   self.weights = weights
	end

end

function DICECriterion:updateOutput(input, target)

	assert(input:nElement() == target:nElement(), "input and target size mismatch")

	local weights = self.weights

	local numerator, denom, output, common

	numerator = input.new()

	numerator:resizeAs(input)

	if weights ~= nil and target:dim() ~= 1 then
	      weights = self.weights:view(1, target:size(2)):expandAs(target)
	end

	numerator:add(input, eps)

	--	compute numerator: 2 * |X n Y|   ; eps for numeric stability
	common = torch.eq(numerator, target)  --find logical equivalence between both
	numerator = 2*torch.sum(common)

	-- compute denominator: |X| + |Y|
	denom = input:nElement() + target:nElement() + eps

	local output = numerator / denom
	self.output = output

	return self.output
end

function DICECriterion:updateGradInput(input, target)

	assert(input:nElement() == target:nElement(), "inputs and target size mismatch")

	--[[ 
			2 * |X| * |Y|   
	    -------------------
	   	 |X|*(|X| + |Y|)^2
	    ]]

	local weights = self.weights
	local gradInput = self.gradInput or input.new()
	local numerator, denom, den_term2, output

	if weights ~= nil and target:dim() ~= 1 then
	    weights = self.weights:view(1, target:size(2)):expandAs(target)
	end

	gradInput:resizeAs(input)

	gradInput = -1*self:updateOutput(input, target)

	if weights ~= nil then
	    gradInput:cmul(weights)
	end

	if self.sizeAverage then
	    gradInput:div(target:nElement())
	end

	--	2 * |X| * |Y|   
	numerator = 2 * input:nElement() * target:nElement()

	--|X|
	denom = input:nElement()

	--(|X| + |Y|)
	den_term2 = input:nElement() + target:nElement()

	-- |X| * (|X| + |Y|)^2
	denom = denom * (den_term2 * den_term2)

	--compute gradients
	gradInput = numerator / denom

	self.gradInput = gradInput

	return self.gradInput
end

-- function DICECriterion:accGradParameters(input, gradOutput)
-- end

-- function DICECriterion:reset()
-- end