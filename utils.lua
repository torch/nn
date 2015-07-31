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

function nn.utils.recursiveFill(t2, val)
   if torch.type(t2) == 'table' then
      for key,_ in pairs(t2) do
         t2[key] = nn.utils.recursiveFill(t2[key], val)
      end
   elseif torch.isTensor(t2) then
      t2:fill(val)
   else
      error("expecting tensor or table thereof. Got "
           ..torch.type(t2).." instead")
   end
   return t2
end

function nn.utils.recursiveAdd(t1, val, t2)
   if not t2 then
      assert(val, "expecting at least two arguments")
      t2 = val
      val = 1
   end
   val = val or 1
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = nn.utils.recursiveAdd(t1[key], val, t2[key])
      end
   elseif torch.isTensor(t2) and torch.isTensor(t2) then
      t1:add(val, t2)
   else
      error("expecting nested tensors or tables. Got "..
            torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function nn.utils.addSingletonDimension(t, dim)
  assert(torch.isTensor(t), "input tensor expected")
  local dim = dim or 1
  assert(dim > 0 and dim <= (t:dim() + 1), "invalid dimension: " .. dim
             .. '. Tensor is of ' .. t:dim() .. ' dimensions.')

  local view = t.new()
  local size = torch.LongStorage(t:dim() + 1)
  local stride = torch.LongStorage(t:dim() + 1)

  for d = 1, dim - 1 do
    size[d] = t:size(d)
    stride[d] = t:stride(d)
  end
  size[dim] = 1
  stride[dim] = 1
  for d = dim + 1, t:dim() + 1 do
    size[d] = t:size(d - 1)
    stride[d] = t:stride(d - 1)
  end

  view:set(t:storage(), t:storageOffset(), size, stride)
  return view
end


table.unpack = table.unpack or unpack
