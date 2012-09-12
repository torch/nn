local SpatialConvolutionMap, parent = torch.class('nn.SpatialConvolutionMap', 'nn.Module')

nn.tables = nn.tables or {}

function nn.tables.full(nin, nout)
   local ft = torch.Tensor(nin*nout,2)
   local p = 1
   for j=1,nout do
      for i=1,nin do
	 ft[p][1] = i
	 ft[p][2] = j
	 p = p + 1
      end
   end
   return ft
end

function nn.tables.oneToOne(nfeat)
   local ft = torch.Tensor(nfeat,2)
   for i=1,nfeat do
      ft[i][1] = i
      ft[i][2] = i
   end
   return ft
end

function nn.tables.random(nin, nout, nto)
   local nker = nto * nout
   local tbl = torch.Tensor(nker, 2)
   local fi = torch.randperm(nin)
   local frcntr = 1
   local tocntr = 1
   local nfi = math.floor(nin/nto) -- number of distinct nto chunks 
   local rfi = math.mod(nin,nto) -- number of remaining from maps
   local totbl = tbl:select(2,2)
   local frtbl = tbl:select(2,1)
   local fitbl = fi:narrow(1, 1, (nfi * nto)) -- part of fi that covers distinct chunks
   local ufrtbl= frtbl:unfold(1, nto, nto)
   local utotbl= totbl:unfold(1, nto, nto)
   local ufitbl= fitbl:unfold(1, nto, nto)
   
   -- start filling frtbl
   for i=1,nout do -- fro each unit in target map
      ufrtbl:select(1,i):copy(ufitbl:select(1,frcntr))
      frcntr = frcntr + 1
      if frcntr-1 ==  nfi then -- reset fi
	 fi:copy(torch.randperm(nin))
	 frcntr = 1
      end
   end
   for tocntr=1,utotbl:size(1) do
      utotbl:select(1,tocntr):fill(tocntr)
   end
   return tbl
end

function constructTableRev(conMatrix)
   local conMatrixL = conMatrix:type('torch.LongTensor')
   -- Construct reverse lookup connection table
   local thickness = conMatrixL:select(2,2):max()
   -- approximate fanin check
   if (#conMatrixL)[1] % thickness == 0 then 
      -- do a proper fanin check and set revTable
      local fanin = (#conMatrixL)[1] / thickness
      local revTable = torch.Tensor(thickness, fanin, 2)
      for ii=1,thickness do
	 local tempf = fanin
	 for jj=1,(#conMatrixL)[1] do
	    if conMatrixL[jj][2] == ii then
	       if tempf <= 0 then break end
	       revTable[ii][tempf][1] = conMatrixL[jj][1]
	       revTable[ii][tempf][2] = jj
	       tempf = tempf - 1
	    end
	 end
	 if tempf ~= 0 then 
	    fanin = -1
	    break
	 end
      end
      if fanin ~= -1 then
	 return revTable
      end
   end
   return {}
end

function SpatialConvolutionMap:__init(conMatrix, kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH
   self.connTable = conMatrix
   self.connTableRev = constructTableRev(conMatrix)
   self.nInputPlane = self.connTable:select(2,1):max()
   self.nOutputPlane = self.connTable:select(2,2):max()
   self.weight = torch.Tensor(self.connTable:size(1), kH, kW)
   self.bias = torch.Tensor(self.nOutputPlane)
   self.gradWeight = torch.Tensor(self.connTable:size(1), kH, kW)
   self.gradBias = torch.Tensor(self.nOutputPlane)
   
   self:reset()
end

function SpatialConvolutionMap:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
      self.weight:apply(function()
			   return torch.uniform(-stdv, stdv)
			end)
      self.bias:apply(function()
			 return torch.uniform(-stdv, stdv)
		      end)
   else
      local ninp = torch.Tensor(self.nOutputPlane):zero()
      for i=1,self.connTable:size(1) do ninp[self.connTable[i][2]] =  ninp[self.connTable[i][2]]+1 end
      for k=1,self.connTable:size(1) do
	 stdv = 1/math.sqrt(self.kW*self.kH*ninp[self.connTable[k][2]])
	 self.weight:select(1,k):apply(function() return torch.uniform(-stdv,stdv) end)
      end
      for k=1,self.bias:size(1) do
	 stdv = 1/math.sqrt(self.kW*self.kH*ninp[k])
	 self.bias[k] = torch.uniform(-stdv,stdv)
      end
   end
end

function SpatialConvolutionMap:updateOutput(input)
   input.nn.SpatialConvolutionMap_updateOutput(self, input)
   return self.output
end

function SpatialConvolutionMap:updateGradInput(input, gradOutput)
   input.nn.SpatialConvolutionMap_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function SpatialConvolutionMap:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialConvolutionMap_accGradParameters(self, input, gradOutput, scale)
end

function SpatialConvolutionMap:decayParameters(decay)
   self.weight:add(-decay, self.weight)
   self.bias:add(-decay, self.bias)
end
