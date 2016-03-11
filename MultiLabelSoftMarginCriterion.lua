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
--
--]]


local MultiLabelSoftMarginCriterion, parent =
torch.class('nn.MultiLabelSoftMarginCriterion', 'nn.Criterion')


function MultiLabelSoftMarginCriterion:__init(weights)
    parent.__init(self)
    self.lsm = nn.Sigmoid()
    self.nll = nn.BCECriterion(weights)
end

function MultiLabelSoftMarginCriterion:updateOutput(input, target)
    input = input:nElement() == 1 and input or input:squeeze()
    target = target:nElement() == 1 and target or target:squeeze()
    self.lsm:updateOutput(input)
    self.nll:updateOutput(self.lsm.output, target)
    self.output = self.nll.output
    return self.output
end

function MultiLabelSoftMarginCriterion:updateGradInput(input, target)
    local size = input:size()
    input = input:nElement() ==1 and input or input:squeeze()
    target = target:nElement() == 1 and target or target:squeeze()
    self.nll:updateGradInput(self.lsm.output, target)
    self.lsm:updateGradInput(input, self.nll.gradInput)
    self.gradInput:view(self.lsm.gradInput, size)
    return self.gradInput
end

 return nn.MultiLabelSoftMarginCriterion
