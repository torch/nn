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
            local tensor = t1[k]
            if not tensor:isSameSizeAs(v) then
               t1[k]:resizeAs(v)
               t1[k]:zero()
            end
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
   if self.index < 0 then
      self.gradInput[#input + self.index + 1] = gradOutput
   else
      self.gradInput[self.index] = gradOutput
   end
   zeroTableCopy(self.gradInput, input)

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end

function SelectTable:type(type, tensorCache)
   self.gradInput = {}
   self.output = {}
   return parent.type(self, type, tensorCache)
end
