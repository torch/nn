local SpatialLPPooling, parent = torch.class('nn.SpatialLPPooling', 'nn.Sequential')

function SpatialLPPooling:__init(nInputPlane, pnorm, kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or kW
   dH = dH or kH
   
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.nInputPlane = nInputPlane
   self.learnKernel = learnKernel

   if pnorm == 2 then
      self:add(nn.Square())
   else
      self:add(nn.Power(pnorm))
   end
   self:add(nn.SpatialConvolutionMap(nn.tables.oneToOne(nInputPlane), kW, kH, dW, dH))
   if pnorm == 2 then
      self:add(nn.Sqrt())
   else
      self:add(nn.Power(1/pnorm))
   end

   self:get(2).bias:zero()
   self:get(2).weight:fill(1/(kW*kH))
   self:get(2).accGradParameters = nil
end
