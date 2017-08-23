local AlphaDropout, Parent = torch.class('nn.AlphaDropout', 'nn.Module')

--[[
During training, Dropout masks parts of the input using binary samples 
from a bernoulli distribution. 
Each input element has a probability of p of being dropped.
]]

function AlphaDropout:__init(p)
   Parent.__init(self)
   self.p = p or 0.5
   if self.p >= 1 or self.p < 0 then
      error('<Dropout> illegal percentage, must be 0 <= p < 1')
   end
   self.train = true
   self.alpha = -1.7580993408473766
   self.keep_prob = 1 - self.p
   self.a = (self.keep_prob + self.alpha^2 * self.keep_prob *(1 - self.keep_prob))^(-0.5)
   self.b = -self.a * self.alpha * (1 - self.keep_prob)
   self.noise = torch.Tensor()
end

function AlphaDropout:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if self.p > 0 then
      if self.train then
         self.noise:resizeAs(input)
         self.noise:bernoulli(self.keep_prob)
         self.output:maskedFill(torch.lt(self.noise, 1), self.alpha)
         self.output:mul(self.a):add(self.b)
      end
   end
   return self.output
end

function AlphaDropout:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   if self.train then
      if self.p > 0 then
         self.gradInput:cmul(self.noise):mul(self.a)
      end
   end
   return self.gradInput
end

function AlphaDropout:setp(p)
   self.p = p
end

function AlphaDropout:__tostring__()
   return string.format('%s(%f)', torch.type(self), self.p)
end


function AlphaDropout:clearState()
   if self.noise then
      self.noise:set()
   end
   return Parent.clearState(self)
end