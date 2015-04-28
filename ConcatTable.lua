local ConcatTable, parent = torch.class('nn.ConcatTable', 'nn.Container')

function ConcatTable:__init()
   parent.__init(self)
   self.modules = {}
   self.output = {}
end

function ConcatTable:updateOutput(input)
   for i=1,#self.modules do
      self.output[i] = self.modules[i]:updateOutput(input)
   end
   return self.output
end

local function retable(t1, t2, f)
   for k, v in pairs(t2) do
      if (torch.type(v) == "table") then
         t1[k] = retable(t1[k] or {}, t2[k], f)
      else
         f(t1, k, v)
      end
   end
   return t1
end

function ConcatTable:updateGradInput(input, gradOutput)
   local isTable = torch.type(input) == 'table'
   local wasTable = torch.type(self.gradInput) == 'table'
   if isTable then
      for i,module in ipairs(self.modules) do
         local currentGradInput = module:updateGradInput(input, gradOutput[i])
         if torch.type(currentGradInput) ~= 'table' then
            error"currentGradInput is not a table!"
         end
         if #input ~= #currentGradInput then
            error("table size mismatch: "..#input.." ~= "..#currentGradInput)
         end
         if i == 1 then
            self.gradInput = wasTable and self.gradInput or {}
            retable(self.gradInput, currentGradInput,
               function(t, k, v)
                  t[k] = t[k] or v:clone()
                  t[k]:resizeAs(v)
                  t[k]:copy(v)
               end
            )
         else
            retable(self.gradInput, currentGradInput,
               function(t, k, v)
                  t[k]:add(v)
               end
            )
         end
      end
   else
      self.gradInput = (not wasTable) and self.gradInput or input:clone()
      for i,module in ipairs(self.modules) do
         local currentGradInput = module:updateGradInput(input, gradOutput[i])
         if i == 1 then
            self.gradInput:resizeAs(currentGradInput):copy(currentGradInput)
         else
            self.gradInput:add(currentGradInput)
         end
      end
   end
   return self.gradInput
end

function ConcatTable:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   for i,module in ipairs(self.modules) do
      module:accGradParameters(input, gradOutput[i], scale)
   end
end

function ConcatTable:accUpdateGradParameters(input, gradOutput, lr)
   for i,module in ipairs(self.modules) do
      module:accUpdateGradParameters(input, gradOutput[i], lr)
   end
end

function ConcatTable:zeroGradParameters()
   for _,module in ipairs(self.modules) do
      module:zeroGradParameters()
   end
end

function ConcatTable:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {' .. line .. tab .. 'input'
   for i=1,#self.modules do
      if i == self.modules then
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
      else
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
      end
   end
   str = str .. line .. tab .. last .. 'output'
   str = str .. line .. '}'
   return str
end
