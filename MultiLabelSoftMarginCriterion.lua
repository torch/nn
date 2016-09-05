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
    self.sigmoid = nn.Sigmoid()
end

function MultiLabelSoftMarginCriterion:updateOutput(input, target)
    local input_size = input:size()
    local x = input:view(input:nElement())
    local t = target:view(target:nElement())

    self.sigmoid:updateOutput(x)

    local indicator = x:ge(0):typeAs(x)
    self.output = torch.sum(
        torch.log(1 + torch.exp(x - torch.cmul(x, indicator):mul(2))) 
        - torch.cmul(x, t - indicator)
        )
    
    if self.sizeAverage then
        self.output = self.output / input:nElement()
    end
    
    return self.output
end

function MultiLabelSoftMarginCriterion:updateGradInput(input, target)
    local t = target:view(target:nElement())

    self.gradInput = self.sigmoid.output - t
    self.gradInput:resizeAs(input)

    if self.sizeAverage then
        self.gradInput:div(target:nElement())
    end

    return self.gradInput
end

 return nn.MultiLabelSoftMarginCriterion
