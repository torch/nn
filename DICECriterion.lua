--[[
	A  statistic used for comparing the similarity of two samples based on 
	Sorensen's original formula ==> 	 2 * |X n Y|
	                                	------------
	                               		 |X| + |Y|

	where |X| and |Y| are the numbers of elements in the two samples. 
	The resulting quotient is an indicator measure of the similarity between the two samples.
	It ranges between 0 and 1. If it is 1, the two images are perfectly similar. Otherwise, 
	they are perfectly dissimilar.

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

	local numerator, denom, common

	if weights ~= nil and target:dim() ~= 1 then
	      weights = self.weights:view(1, target:size(2)):expandAs(target)
	end

	-- compute numerator: 2 * |X n Y|   ; eps for numeric stability
	common = torch.eq(input, target)  --find logical equivalence between both
	common:mul(2)
	numerator = torch.sum(common)

	-- compute denominator: |X| + |Y|
	denom = input:nElement() + target:nElement() + eps

	self.output = numerator/denom

	return self.output
end

function DICECriterion:updateGradInput(input, target)

	assert(input:nElement() == target:nElement(), "inputs and target size mismatch")

	--[[ 
								2 * |X| * |Y|   
	    		Gradient = 	 ----------------------
	   	 						|X|*(|X| + |Y|)^2
	    ]]

	local weights = self.weights
	local gradInput = self.gradInput or input.new()
	local numerator, denom, den_term2, output

	gradInput:resizeAs(input)

	if weights ~= nil and target:dim() ~= 1 then
	    weights = self.weights:view(1, target:size(2)):expandAs(target)
	end

	if weights ~= nil then
	    gradInput:cmul(weights)
	end

	if self.sizeAverage then
	    gradInput:div(target:nElement())
	end

	-- compute 2 * |X| * |Y|   
	numerator = 2 * input:nElement() * target:nElement()

	-- compute |X|
	denom = input:nElement()

	-- compute (|X| + |Y|)
	den_term2 = input:nElement() + target:nElement()

	-- compute |X| * (|X| + |Y|)^2
	denom = denom * (den_term2 * den_term2)

	-- compute gradients
	gradInput = numerator / denom

	self.gradInput = gradInput

	return self.gradInput
end