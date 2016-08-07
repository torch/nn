--[[
		Computes the Sorensen-dice coefficient of similarity given two samples. 
		The quotient of similarity is defined as:

   		 	    Q =     2 * (X n Y)
         		     -------------------
        		      sum_i(X) + sum_i(Y)
		where X and Y are the two samples; 
		(X n Y) denote the intersection where the elements of X and Y are equal.

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

	-- compute 2 * (X intersection Y)
	common = torch.eq(input, target)  		--find logical equivalence between both
	numerator = torch.sum(common)
	numerator = numerator * 2

	-- compute denominator: sum_i(X) + sum_i(Y)
	denom = input:nElement() + target:nElement() + eps

	output = numerator/denom

	self.output = -output

	return self.output
end

function DICECriterion:updateGradInput(input, target)
	--[[ 
								      2 * sum_i(X) * sum_i(Y)  
	    		Gradient = 	    ---------------------------------
	   	 						sum_i(X)*(sum_i(X) + sum_i(Y))^2
	    ]]

	assert(input:nElement() == target:nElement(), "inputs and target size mismatch")
	self.buffer = self.buffer or input.new()

	local buffer = self.buffer
	local weights = self.weights
	local gradInput = self.gradInput 

	if weights ~= nil and target:dim() ~= 1 then
	    weights = self.weights:view(1, target:size(2)):expandAs(target)
	end

	buffer:resizeAs(input)
    buffer:zero()

	-- compute sum_i(X) + sum_i(Y) + eps 
	buffer:add(input:nElement()):add(target:nElement()):add(eps)
	-- compute (sum_i(X) + sum_i(Y) + eps )^2 + eps
	buffer:cmul(buffer):add(eps)
	-- compute sum_i(X)*(sum_i(X) + sum_i(Y) + eps )^2 + eps
	buffer:mul(input:nElement())

	gradInput:resizeAs(input)	
	gradInput:zero()

	-- compute 2 * sum_i(X) * sum_i(Y)
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
