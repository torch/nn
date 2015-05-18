nn.utils = {}

function nn.utils.recursiveType(param, type_str)
   if torch.type(param) == 'table' then
      for k, v in pairs(param) do
         param[k] = nn.utils.recursiveType(v, type_str)
      end
   elseif torch.isTypeOf(param, 'nn.Module') or
          torch.isTypeOf(param, 'nn.Criterion') then
      param:type(type_str)
   elseif torch.isTensor(param) then
       param = param:type(type_str)
   end
   return param
end

function nn.utils.recursiveResizeAs(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = nn.utils.recursiveResizeAs(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resizeAs(t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end


table.unpack = table.unpack or unpack
