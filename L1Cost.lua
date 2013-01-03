local L1Cost, parent = torch.class('nn.L1Cost','nn.Criterion')

function L1Cost:__init()
   parent.__init(self)
end

function L1Cost:updateOutput(input)
   return input.nn.L1Cost_updateOutput(self,input)
end

function L1Cost:updateGradInput(input)
   return input.nn.L1Cost_updateGradInput(self,input)
end

