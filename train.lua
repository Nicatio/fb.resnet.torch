--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--
require 'cudnn'
local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, preModel, donModel, chSelector, criterion, opt, optimState)
   self.model = model
   self.preModel = preModel
   self.donModel = donModel
   self.chSelector = chSelector
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = self.model:getParameters()
   if self.opt.preTarget == 'class' then
      self.criterionClass = nn.CrossEntropyCriterion():type(opt.tensorType)
      
      self.optimStateClass = {table.unpack(self.optimState)}
   else 
      self.criterionClass = nil
   end
   print (self.model)
   print ('')
   print (' - weightDecay:   ' .. opt.weightDecay)
   print (' - momentum:      ' .. opt.momentum)
   print (' - learningRate:  ' .. opt.LR)
   print (' - randomCrop:    ' .. tostring(opt.randCrop))
   print (' - retrainOnlyFC: ' .. tostring(opt.retrainOnlyFC))
   print (' - iGPU:          ' .. opt.iGPU)
   print (' - #params:       ' .. self.params:size(1))
   print ('')
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   if self.opt.preModel == 'none' then
      self.optimState.learningRate = self:learningRate(epoch)
   else
      self.optimState.learningRate = self:learningRate(epoch) / 1000
      if self.opt.preTarget == 'class' or self.opt.preTarget == 'hybrid' then
         self.optimStateClass.learningRate = self:learningRate(epoch)
      end
   end
   local timer = torch.Timer()
   local dataTimer = torch.Timer()

--   if epoch >= 200 then
--   print('crit')
--   print('crit2')
--      self.criterion = nn.CrossEntropyCriterion():type(self.opt.tensorType)
--   end
   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
  
   local tempModel = nn.Sequential()
  
   for n, sample in dataloader:run(self.opt.randCrop) do
      local dataTime = dataTimer:time().real
      local output, batchSize, loss
      -- Copy input and target to the GPU
      self:copyInputs(sample)
      if self.opt.preModel == 'none' then
         output = self.model:forward(self.input):float()
         batchSize = output:size(1)
         loss = self.criterion:forward(self.model.output, self.target)
         self.model:zeroGradParameters()
         self.criterion:backward(self.model.output, self.target)
         self.model:backward(self.input, self.criterion.gradInput)
         optim.sgd(feval, self.params, self.optimState)
      else
         if self.opt.preTarget == 'hybrid' or self.opt.preTarget == 'conv' then
           if self.opt.preTarget == 'hybrid' then
              tempModel:add(self.model:get(#self.model.modules-2))
              tempModel:add(self.model:get(#self.model.modules-1))
              tempModel:add(self.model:get(#self.model.modules))
              self.model:remove(#self.model.modules)
              self.model:remove(#self.model.modules)
              self.model:remove(#self.model.modules)
           end
           self.targetConv = self.preModel:forward(self.input)
           if self.opt.chSelector ~= 'none' then
              self.targetConv = self.targetConv:index(2, self.chSelector[{{1,self.opt.nLastLayerCh}}])
           end
  
           output = self.model:forward(self.input):float()
           batchSize = output:size(1)
           loss = self.criterion:forward(self.model.output, self.targetConv)
     
           self.model:zeroGradParameters()
           self.criterion.gradInput = torch.CudaTensor()
           self.criterion.gradInput:resizeAs(self.model.output):zero()
           self.criterion:backward(self.model.output, self.targetConv)
           self.model:backward(self.input, self.criterion.gradInput)
           if self.opt.retrainOnlyFC == true then
               self.gradParams:narrow(1,1,self.gradParams:size()[1]-1320-10):zero()
           end
           optim.sgd(feval, self.params, self.optimState)
           
           if self.opt.preTarget == 'hybrid' then
              self.model:add(tempModel:get(#tempModel.modules-2))
              self.model:add(tempModel:get(#tempModel.modules-1))
              self.model:add(tempModel:get(#tempModel.modules))
              tempModel:remove(#tempModel.modules)
              tempModel:remove(#tempModel.modules)
              tempModel:remove(#tempModel.modules)
              
              output = self.model:forward(self.input):float()
              batchSize = output:size(1)
              loss = self.criterionClass:forward(self.model.output, self.target)
              self.model:zeroGradParameters()
              self.criterionClass:backward(self.model.output, self.target)
              self.model:backward(self.input, self.criterionClass.gradInput)
              optim.sgd(feval, self.params, self.optimStateClass)
           end
         elseif self.opt.preTarget == 'class' then
if epoch <200 then
            self.target = self.preModel:forward(self.input)
            end
            output = self.model:forward(self.input):float()
            batchSize = output:size(1)
            loss = self.criterion:forward(self.model.output, self.target)
            self.model:zeroGradParameters()
            self.criterion.gradInput = torch.CudaTensor()
            self.criterion.gradInput:resizeAs(self.model.output):zero()
            self.criterion:backward(self.model.output, self.target)
            self.model:backward(self.input, self.criterion.gradInput)
            optim.sgd(feval, self.params, self.optimState)
         end
         
      end
      local top1, top5
      lossSum = lossSum + loss*batchSize
      N = N + batchSize
      if self.opt.preModel ~= 'none' and self.opt.preTarget == 'conv' then
         top1 = 0
         top5 = 0
         io.write((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %7.3f (%7.3f)\r'):format(
         epoch, n, trainSize, timer:time().real, dataTime, loss, lossSum / N))
      else 
         top1, top5 = self:computeScore(output, sample.target, 1)
         top1Sum = top1Sum + top1*batchSize
         top5Sum = top5Sum + top5*batchSize
         io.write((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)\r'):format(
         epoch, n, trainSize, timer:time().real, dataTime, loss, top1, top1Sum / N, top5, top5Sum / N))
      end

      -- check that the storage didn't get changed due to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end
   io.write('\n')
   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0
   
   
   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)


      local output, batchSize, loss
      output = self.model:forward(self.input):float()
      batchSize = output:size(1) / nCrops
      if self.opt.preModel == 'none' then
         loss = self.criterion:forward(self.model.output, self.target)
      else
         if self.opt.preTarget == 'conv' then
            self.target = self.preModel:forward(self.input)
            if self.opt.chSelector ~= 'none' then
               self.target = self.target:index(2, self.chSelector[{{1,self.opt.nLastLayerCh}}])
            end
            loss = self.criterion:forward(self.model.output, self.target) 
         else
            loss = self.criterionClass:forward(self.model.output, self.target)
         end
      end
      local top1, top5
      if self.opt.preModel ~= 'none' and self.opt.preTarget == 'conv' then
         lossSum = lossSum + loss*batchSize
         N = N + batchSize
         io.write((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  loss %7.3f (%7.3f)\r'):format(
         epoch, n, size, timer:time().real, dataTime, loss, lossSum / N))
         
         top1Sum = lossSum
         top5Sum = lossSum
      else 
         top1, top5 = self:computeScore(output, sample.target, nCrops)
         top1Sum = top1Sum + top1*batchSize
         top5Sum = top5Sum + top5*batchSize
         N = N + batchSize
         io.write((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)\r'):format(
         epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N))
         
      end
      
      timer:reset()
      dataTimer:reset()
   end
   io.write('\n')
   self.model:training()
   if self.opt.preModel ~= 'none' and self.opt.preTarget == 'conv' then
      print((' * Finished epoch # %d     lr %e     loss %7.3f\n'):format(
         epoch, self:learningRate(epoch-1), lossSum / N))
   else
      print((' * Finished epoch # %d     lr %e     top1: %7.3f     top5: %7.3f\n'):format(
         epoch, self:learningRate(epoch-1), top1Sum / N, top5Sum / N))
   end
   return top1Sum / N, top5Sum / N
end

function Trainer:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():topk(5, 2, true, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(predictions))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

local function getCudaTensorType(tensorType)
  if tensorType == 'torch.CudaHalfTensor' then
     return cutorch.createCudaHostHalfTensor()
  elseif tensorType == 'torch.CudaDoubleTensor' then
    return cutorch.createCudaHostDoubleTensor()
  else
     return cutorch.createCudaHostTensor()
  end
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch[self.opt.tensorType:match('torch.(%a+)')]()
      or getCudaTensorType(self.opt.tensorType))
   self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor())
   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if string.sub(self.opt.netType,1,8) == 'densenet' then
      if self.opt.dataset == 'imagenet' then
         decay = math.floor((epoch - 1) / 30)
      elseif self.opt.dataset == 'cifar10' then
         --decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
         if self.opt.preModel ~= 'none' then
            decay = epoch >= 225 and 2 or epoch >= 150 and 1 or 0
         elseif self.opt.donModel ~= 'none' then
            --decay =  epoch >= 175 and 3 or epoch >= 100 and 2 or epoch >= 25 and 1 or 0
            --decay = epoch >= 150 and 3 or epoch >= 75 and 2 or 1
            --decay = epoch >= 225 and 2 or epoch >= 150 and 1 or 0
            decay = epoch >= 375 and 3 or epoch >= 300 and 2 or epoch >= 150 and 1 or 0
            --decay = epoch >= 375 and 3 or epoch >=300 and 2 or epoch >= 225 and 1.5 or epoch >=150 and 1 or 0
            --decay = epoch >= 225 and 2 or epoch >= 150 and 1 or 0
            --decay = epoch >= 225 and 3 or epoch >= 150 and 2 or epoch >= 75 and 1 or 0
         else
            decay = epoch >= 225 and 2 or epoch >= 150 and 1 or 0
         end
      elseif self.opt.dataset == 'cifar100' then
         decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
      end
      return self.opt.LR * math.pow(0.1, decay)
   elseif string.sub(self.opt.netType,1,10) == 'wideresnet' then
      if self.opt.dataset == 'imagenet' then
         decay = math.floor((epoch - 1) / 30)
         return self.opt.LR * math.pow(0.1, decay)
      elseif self.opt.dataset == 'cifar10' then
--         decay = epoch >= 160 and 3 or epoch >= 120 and 2 or epoch >= 60 and 1 or 0
         decay = epoch >= 400 and 4 or epoch >= 300 and 3 or epoch >= 200 and 2 or epoch >= 100 and 1 or 0
         --decay = epoch >= 500 and 3 or epoch >= 400 and 2 or epoch >= 200 and 1 or 0
         --decay = epoch >= 500 and 3 or epoch >= 400 and 2 or epoch >= 300 and 1 or 0
         return self.opt.LR * math.pow(0.2, decay)
      elseif self.opt.dataset == 'cifar100' then
         decay = epoch >= 120 and 2 or epoch >= 80 and 1 or 0
         return self.opt.LR * math.pow(0.1, decay)
      end
   else
      if self.opt.dataset == 'imagenet' then
         decay = math.floor((epoch - 1) / 30)
      elseif self.opt.dataset == 'cifar10' then
         decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
      elseif self.opt.dataset == 'cifar100' then
         decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
      end
      return self.opt.LR * math.pow(0.1, decay)
   end
end

function Trainer:featureExtract(epoch, dataloader, layerIndex)
   -- Extracts intermediate features

   local dataSize = dataloader:dataSize()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0
   local extracted
   self.model:evaluate()
   for n, sample in dataloader:run() do

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1) / nCrops
      local loss = self.criterion:forward(self.model.output, self.target)
      local res = self.model:get(layerIndex).output:squeeze()
      if n == 1 then
         local esize = res:size()
         esize[1] = dataSize
         extracted = torch.Tensor(esize):type(res:type())
      end
      extracted:narrow(1, N+1, batchSize):copy(res:narrow(1, 1, batchSize))
      io.write((' | Extract: [%d/%d]\r'):format(n, size))
      N = N + batchSize
   end
   print('\nFinished.')
   
   return extracted
end

return M.Trainer
