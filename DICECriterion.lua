--[[
	Author: Olalekan Ogunmolu, July 2016
			patlekano@gmail.com
]]

local DICECriterion, parent = torch.class('nn.DICECriterion', 'nn.Criterion')

local eps = 1

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

	local numerator, denom, common, output

	if weights ~= nil and target:dim() ~= 1 then
	      weights = self.weights:view(1, target:size(2)):expandAs(target)
	end

	-- compute numerator: 2 * |X n Y|   ; eps for numeric stability
	common = torch.eq(input, target)  --find logical equivalence between both
	numerator = torch.sum(common)
	numerator = numerator * 2

	-- compute denominator: |X| + |Y|
	denom = input:nElement() + target:nElement() + eps

	output = numerator/denom

	self.output = -output

	return self.output
end

function DICECriterion:updateGradInput(input, target)
	--[[ 
								2 * |X| * |Y|   
	    		Gradient = 	 ----------------------
	   	 						|X|*(|X| + |Y|)^2
	    ]]

	assert(input:nElement() == target:nElement(), "inputs and target size mismatch")
	self.buffer = self.buffer or input.new()

	local buffer = self.buffer
	local weights = self.weights
	local gradInput = self.gradInput --or input.new()

	if weights ~= nil and target:dim() ~= 1 then
	    weights = self.weights:view(1, target:size(2)):expandAs(target)
	end

	buffer:resizeAs(input)
	-- compute |X| + |Y| + eps 
	buffer:add(input:nElement()):add(target:nElement()):add(eps)
	-- compute (|X| + |Y| + eps)^2 + eps
	buffer:cmul(buffer):add(eps)
	-- compute |X|(|X| + |Y| + eps)^2 + eps
	buffer:mul(input:nElement())

	gradInput:resizeAs(input)	
	-- compute 2 * |X| * |Y|  
	gradInput:add(input:nElement()):mul(target:nElement()):mul(2)

	-- compute quotient
	gradInput:cdiv(buffer)

	if weights ~= nil then
	    gradInput:cmul(weights)
	end

	if self.sizeAverage then
	    gradInput:div(target:nElement())
	end

	return gradInput
end

function DICECriterion:accGradParameters(input, gradOutput)
end

function DICECriterion:reset()
end