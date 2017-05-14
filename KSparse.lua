local k_sparse, parent = torch.class('nn.KSparse','nn.Module')

function table.contains(table, element)
  for _, value in pairs(table) do
    if value == element then
      return true
    end
  end
  return false
end

function k_sparse:__init(k)
    parent.__init(self)
    self.k = k
	self.indexes=0
end
    
function k_sparse:updateOutput(input)
    local m = self.k
	local mask
	local input_temp

	res, ind = input:topk(m,true)
	local top_ind_table={}
	local top_ind=0
	local top_frequency=0
	if (ind:dim()==2) then
		for i=1, ind:size(2) do
			for j=1, ind:size(1) do
				frequency = ind[{{},{i}}]:eq(ind[j][i]):sum()
				if (frequency>top_frequency) then
					top_frequency=frequency
					top_ind=ind[j][i]
				end
			end
			table.insert(top_ind_table, top_ind)
			top_ind=0
			top_frequency=0
		end
	end

	if (ind:dim()==1) then
		top_ind_table=torch.totable(ind)
	end
	local ind_t=top_ind_table
	if (input:dim()==2) then
		mask=torch.CudaTensor(input:size(1),input:size(2)):zero()
		for k, v in pairs(ind_t) do
			mask[{{},{v}}]:fill(1)
		end
	end
	if (input:dim()==1) then
		mask=torch.CudaTensor(input:size(1)):zero()
		for k, v in pairs(ind_t) do
			mask[{{v}}]:fill(1)
		end
	end 
	self.indexes=ind_t
	self.output = torch.cmul(input,mask)
    return self.output 
end

function k_sparse:updateGradInput(input, gradOutput)
	local ind_t=self.indexes
	local mask
	if (gradOutput:dim()==2) then
		mask=torch.CudaTensor(gradOutput:size(1),gradOutput:size(2)):zero()
		for k, v in pairs(ind_t) do
			mask[{{},{v}}]:fill(1)
		end
	end
	if (gradOutput:dim()==1) then
		mask=torch.CudaTensor(gradOutput:size(1)):zero()
		for k, v in pairs(ind_t) do
			mask[{{v}}]:fill(1)
		end
	end
	self.gradInput=torch.cmul(gradOutput,mask)
	collectgarbage()
    return self.gradInput
end