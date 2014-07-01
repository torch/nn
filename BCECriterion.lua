local BCECriterion, parent = torch.class('nn.BCECriterion', 'nn.Criterion')

function BCECriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function BCECriterion:updateOutput(input, target)
    -- log(input) * target + log(1 - input) * (1 - target)

    self.term1 = self.term1 or input.new()
    self.term2 = self.term2 or input.new()
    self.term3 = self.term3 or input.new()

    self.term1:resizeAs(input)
    self.term2:resizeAs(input)
    self.term3:resizeAs(input)

    self.term1:fill(1):add(-1,target)
    self.term2:fill(1):add(-1,input):log():cmul(self.term1)
    
    self.term3:copy(input):log():cmul(target)
    self.term3:add(self.term2)

    if self.sizeAverage then
    	self.term3:div(target:size(1))
    end

    return self.term3:sum()
end

function BCECriterion:updateGradInput(input, target)
    -- target / input - (1 - target) / (1 - input)
    
    self.term1 = self.term1 or input.new()
    self.term2 = self.term2 or input.new()
    
    self.term1:resizeAs(input)
    self.term2:resizeAs(input)
    
    self.term1:fill(1):add(-1,target)
    self.term2:fill(1):add(-1,input)

    self.term1:cdiv(self.term2)

    self.gradInput:resizeAs(input)
    self.gradInput:copy(target):cdiv(input)

    self.gradInput:add(-1,self.term1)

    if self.sizeAverage then
    	self.gradInput:div(target:size(1))
    end
    
    return self.gradInput
end
