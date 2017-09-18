--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'sys'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'
require 'DropC'
require 'DropCS'
require 'DropCL'
--require 'DropFlip'
require 'SELU'
require 'AlphaDropout'
--
--require 'DropCm'
--require 'DropChannel'
--require 'DropCStd'
--require 'DropCStd2'
--require 'DropCMean'
--require 'DropCMean2'
--require 'SpatialConvolutionDrop'
--require 'BND'

-- we don't  change this to the 'correct' type (e.g. HalfTensor), because math
-- isn't supported on that type.  Type conversion later will handle having
-- the correct type.
torch.setdefaulttensortype('torch.FloatTensor')
--torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)
cutorch.setDevice(opt.iGPU)

-- Load previous checkpoint, if it exists
local checkpoint, optimState
local preModel
if opt.testOnly then
   checkpoint, optimState = checkpoints.best(opt)
else
   checkpoint, optimState = checkpoints.latest(opt)
end

-- Create model
local model, criterion, preModel, donModel, chSelector = models.setup(opt, checkpoint)
--local params_, gradParams_ = model:getParameters()
--   
--   print (' - #params:       ' .. params_:size(1))
--prin()
-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, preModel, donModel, chSelector, criterion, opt, optimState)

if opt.lsuv then
   local lsuv = require 'lsuv'
   model:lsuvInit(opt)
end

if opt.impInit then
   local impInit = require 'impInit'
   model:impInit(opt, donModel)
end

if opt.testOnly then
   if opt.trTest then
      local top1Err, top5Err = trainer:test(0, trainLoader)
      print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
      local logfile = io.open(('%s/log_%d.txt'):format(opt.save,math.ceil(sys.clock())), 'w')
      logfile:write(('%7.3f\t%7.3f\n'):format(
            top1Err, top5Err))
      logfile:close()
      return
   else
      local top1Err, top5Err = trainer:test(0, valLoader)
      print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
      local logfile = io.open(('%s/log_%d.txt'):format(opt.save,math.ceil(sys.clock())), 'w')
      logfile:write(('%7.3f\t%7.3f\n'):format(
            top1Err, top5Err))
      logfile:close()
      return
   end
end


if opt.feOnly then
   local feLayerIndex = opt.feLayerIndex
   if feLayerIndex < 0 then
      feLayerIndex = #model.modules
   end 

   local feFile = 'feature_' .. feLayerIndex .. '.t7'
   local features = trainer:featureExtract(0, trainLoader, feLayerIndex)

   torch.save(paths.concat(opt.feDir, feFile), features)
   print ('Features saved at ' .. paths.concat(opt.feDir, feFile))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge

if startEpoch == 1 then
   checkpoints.saveInit(model, optimState, opt)
end

for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testTop1, testTop5 = trainer:test(epoch, valLoader)

   local bestModel = false
   if testTop1 < bestTop1 then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      
   end
   print(' * Best model ', bestTop1, bestTop5)
   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
   
   local logfile = io.open(paths.concat(opt.save, 'log.txt'), 'a+')
   logfile:write(('%d\t%e\t%7.3f\t%7.3f\t%7.3f\t%7.3f\t%7.3f\t%7.3f\n'):format(
         epoch, trainer:learningRate(epoch-1), trainTop1, trainTop5, testTop1, testTop5, bestTop1, bestTop5))
   logfile:close()
end

print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))
