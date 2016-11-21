--[[
   Probabilistic Criterion for Triplet Siamese Model for learning embedding.
   Ref: https://arxiv.org/pdf/1412.6622v3.pdf

   loss = -log( exp(-X) / ( exp(-X) + exp(-Y) ) )
   where
   X : Distance between similar samples
   Y : Distance between dissimilar samples

   The loss could be break down to following log expansion

   loss = -log( exp(-X) ) - (-log( exp(-X) + exp(-Y) ))
   loss = -log( exp(-X) ) + log( exp(-X) + exp(-Y) )
   loss = -(-X) + log( exp(-X) + exp(-Y) )
   loss = X + log( exp(-X) + exp(-Y) )
--]]

local DistanceRatioCriterion, parent = torch.class('nn.DistanceRatioCriterion',
                                                   'nn.Criterion')

function DistanceRatioCriterion:__init(sizeAverage)
   parent.__init(self)
   if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = true
   end
end

-- X : Distance between similar samples
-- Y : Distance between dissimilar samples
function DistanceRatioCriterion:updateOutput(input)
   assert(#input == 2, "Invalid number of inputs")
   
   local X = input[1]
   local Y = input[2]

   local noOfDistances = X:size(1)

   assert(X:nElement() == Y:nElement(), "Number of distances don't match.")
   assert(X:size(1) == Y:size(1), "Invalid distances' size.")

   -- Compute exp(-X) and exp(-Y)
   self.expMinusX = self.expMinusX or X.new()
   self.expMinusY = self.expMinusY or Y.new()

   -- Compute ( exp(-X) + exp(-Y) )
   self.expMinusX:resizeAs(X):copy(X):mul(-1):exp()
   self.expMinusY:resizeAs(Y):copy(Y):mul(-1):exp()

   self.sumExpMinusXY = self.sumExpMinusExp or X.new()
   self.sumExpMinusXY:resizeAs(self.expMinusX):copy(self.expMinusX)
                     :add(self.expMinusY)

   -- Compute log( exp(-X) + exp(-Y) )
   self.logSumExpMinusXY = self.logSumExpMinusXY or self.sumExpMinusXY.new()
   self.logSumExpMinusXY:resizeAs(self.sumExpMinusXY):copy(self.sumExpMinusXY)

   -- Compute log( exp(-X) + exp(-Y) )
   self.loss = self.loss or self.logSumExpMinusXY.new()
   self.loss:resizeAs(X):copy(X):add(self.logSumExpMinusXY)

   if self.sizeAverage then
      return self.loss:sum()/noOfDistances
   else
      return self.loss:sum()
   end
end
