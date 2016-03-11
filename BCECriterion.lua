local BCECriterion, parent = torch.class('nn.BCECriterion', 'nn.Criterion')

local eps = 1e-12

function BCECriterion:__init(weights, sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end
    if weights ~= nil then
        assert(weights:dim() == 1, "weights input should be 1-D Tensor")
        self.weights = weights
    end
end


function BCECriterion:__len()
    if (self.weights) then
        return #self.weights
    else
        return 0
    end
end

function BCECriterion:updateOutput(input, target)
    -- - log(input) * target - log(1 - input) * (1 - target)

    assert( input:nElement() == target:nElement(),
    "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    local buffer = self.buffer
    local weights = self.weights
    local output

    buffer:resizeAs(input)

    if weights ~= nil and target:dim() ~= 1 then
        weights = self.weights:view(1, target:size(2)):expandAs(target)
    end

    -- log(input) * target
    buffer:add(input, eps):log()
    if weights ~= nil then buffer:cmul(weights) end

    output = torch.dot(target, buffer)

    -- log(1 - input) * (1 - target)
    buffer:mul(input, -1):add(1):add(eps):log()
    if weights ~= nil then buffer:cmul(weights) end

    output = output + torch.sum(buffer)
    output = output - torch.dot(target, buffer)

    if self.sizeAverage then
        output = output / input:nElement()
    end

    self.output = - output

    return self.output
end

function BCECriterion:updateGradInput(input, target)
    -- - (target - input) / ( input (1 - input) )
    -- The gradient is slightly incorrect:
    -- It should have be divided by (input + eps) (1 - input + eps)
    -- but it is divided by input (1 - input + eps) + eps
    -- This modification requires less memory to be computed.

    assert( input:nElement() == target:nElement(),
    "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    local buffer = self.buffer
    local weights = self.weights
    local gradInput = self.gradInput

    if weights ~= nil and target:dim() ~= 1 then
        weights = self.weights:view(1, target:size(2)):expandAs(target)
    end

    buffer:resizeAs(input)
    -- - x ( 1 + eps -x ) + eps
    buffer:add(input, -1):add(-eps):cmul(input):add(-eps)

    gradInput:resizeAs(input)
    -- y - x
    gradInput:add(target, -1, input)
    -- - (y - x) / ( x ( 1 + eps -x ) + eps )
    gradInput:cdiv(buffer)

    if weights ~= nil then
        gradInput:cmul(weights)
    end

    if self.sizeAverage then
        gradInput:div(target:nElement())
    end

    return gradInput
end
