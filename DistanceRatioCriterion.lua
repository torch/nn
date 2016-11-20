--[[
   Probabilistic Criterion for Triplet Siamese Model for learning embedding.
   Ref: https://arxiv.org/pdf/1412.6622v3.pdf
--]]

local DistanceRatioCriterion, parent = torch.class('nn.DistanceRatioCriterion',
                                                   'nn.Criterion')

function DistanceRatioCriterion:__init()
   parent.__init(self)
   if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = true
   end
end

-- X : Distance between similar samples
-- Y : Distance between dissimilar samples
function DistanceRatioCriterion:updateOutput(X, Y)

   assert(X.nElement() == Y.nElement())

   self.expX = self.expX or X.new()
   self.expY = self.expY or Y.new()
   
end

