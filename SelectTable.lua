local SelectTable, parent = torch.class('nn.SelectTable', 'nn.Module')

function SelectTable:__init(index)
   parent.__init(self)
   self.index = index
   self.gradInput = {}
end

function SelectTable:updateOutput(input)
   self.output = input[self.index]
   return self.output
end

function SelectTable:updateGradInput(input, gradOutput)
   if #self.gradInput == 0 then
      local function zeroTableCopy(t1, t2)
         for k, v in pairs(t2) do
            if (torch.type(v) == "table") then
               t1[k] = zeroTableCopy(t1[k] or {}, t2[k])
            else
               t1[k] = v:clone():zero()
            end
         end
         return t1
      end
      zeroTableCopy(self.gradInput, input)
   end
   self.gradInput[self.index] = gradOutput
   return self.gradInput
end

function SelectTable:type(type)
   self.gradInput = {}
end
