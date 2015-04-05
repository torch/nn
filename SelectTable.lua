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

local function zeroTableCopy(t1, t2)
   for k, v in pairs(t2) do
      if (torch.type(v) == "table") then
         t1[k] = zeroTableCopy(t1[k] or {}, t2[k])
      else
         if not t1[k] then
            t1[k] = v:clone():zero()
         else
            local tensor = t1[k]
            if not tensor:isSameSizeAs(v) then
               t1[k]:resizeAs(v)
               t1[k]:zero()
            end
         end
      end
   end
   return t1
end

function SelectTable:updateGradInput(input, gradOutput)
   self.gradInput[self.index] = gradOutput
   zeroTableCopy(self.gradInput, input)
   return self.gradInput
end

function SelectTable:type(type)
   self.gradInput = {}
   self.output = {}
   return parent.type(self, type)
end
