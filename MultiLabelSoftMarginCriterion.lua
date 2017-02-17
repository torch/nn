--[[
-- A MultiLabel multiclass criterion based on sigmoid:
--
-- the loss is:
-- l(x,y) = - sum_i y[i] * log(p[i]) + (1 - y[i]) * log (1 - p[i])
-- where p[i] = exp(x[i]) / (1 + exp(x[i]))
--
-- and with weights:
-- l(x,y) = - sum_i weights[i] (y[i] * log(p[i]) + (1 - y[i]) * log (1 - p[i]))
--
-- This uses the stable form of the loss and gradients.
--]]


local MultiLabelSoftMarginCriterion, parent =
torch.class('nn.MultiLabelSoftMarginCriterion', 'nn.Criterion')


function MultiLabelSoftMarginCriterion:__init(weights, sizeAverage)
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
    self.sigmoid = nn.Sigmoid()
end

function MultiLabelSoftMarginCriterion:updateOutput(input, target)
    local input_size = input:size()
    local weights = self.weights
    if weights ~= nil and target:dim() ~= 1 then
        weights = self.weights:view(1, target:size(2)):expandAs(target)
    end

    local x = input:view(input:nElement())
    local t = target:view(target:nElement())
    
    self.sigmoid:updateOutput(x)

    local indicator = x:ge(0):typeAs(x)
    local buffer = torch.log(1 + torch.exp(x - torch.cmul(x, indicator):mul(2))) 
        - torch.cmul(x, t - indicator)

    if weights ~= nil then
        buffer:cmul(weights:resize(weights:nElement()))
    end

    self.output = torch.sum(buffer)

    if self.sizeAverage then
        self.output = self.output / input:nElement()
    end
    
    return self.output
end

function MultiLabelSoftMarginCriterion:updateGradInput(input, target)
    local weights = self.weights
    if weights ~= nil and target:dim() ~= 1 then
        weights = self.weights:view(1, target:size(2)):expandAs(target)
    end

    local t = target:view(target:nElement())

    self.gradInput = self.sigmoid.output - t
    self.gradInput:resizeAs(input)

    if weights ~= nil then
        self.gradInput:cmul(weights)
    end

    if self.sizeAverage then
        self.gradInput:div(target:nElement())
    end
    
    return self.gradInput
end

 return nn.MultiLabelSoftMarginCriterion
