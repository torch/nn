local BCECriterion, parent = torch.class('nn.BCECriterion', 'nn.Criterion')

function BCECriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function BCECriterion:updateOutput(input, target)
    -- log(input) * target + log(1 - input) * (1 - target)
    self.output = torch.log(input):cmul(target)
    
    self.output:add(torch.add(-input,1):log():cmul(torch.add(-target,1)))

    if self.sizeAverage
    	self.output:div(target:size(1))
    end

    return self.output:sum()
end

function BCECriterion:updateGradInput(input, target)
    -- target / input - (1 - target) / (1 - input)
    self.gradInput = torch.cdiv(target,input)
    self.gradInput:add(-1,torch.cdiv(torch.add(-target,1),torch.add(-input,1)))

    if self.sizeAverage
    	self.gradInput:div(target:size(1))
    end
    
    return self.gradInput
end
