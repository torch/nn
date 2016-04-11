local SelectTable, parent = torch.class('nn.SelectTable', 'nn.Module')

function SelectTable:__init(index)
   parent.__init(self)
   self.index = index
   self.gradInput = {}
end

function SelectTable:updateOutput(input)
   assert(math.abs(self.index) <= #input, "arg 1 table idx out of range")
   if self.index < 0 then
      self.output = input[#input + self.index + 1]
   else
      self.output = input[self.index]
   end

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
            t1[k]:resizeAs(v)
            t1[k]:zero()
         end
      end
   end
   for k, v in pairs(t1) do
      if not t2[k] then
         t1[k] = nil
      end
   end
   return t1
end

function SelectTable:updateGradInput(input, gradOutput)
   -- make gradInput a zeroed copy of input
   zeroTableCopy(self.gradInput, input)
   -- handle negative indices
   local index = self.index < 0 and #input + self.index + 1 or self.index
   -- copy into gradInput[index] (necessary for variable sized inputs)
   assert(self.gradInput[index])
   nn.utils.recursiveCopy(self.gradInput[index], gradOutput)

   return self.gradInput
end

function SelectTable:type(type, tensorCache)
   self.gradInput = {}
   self.output = {}
   return parent.type(self, type, tensorCache)
end
