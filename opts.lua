--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 ResNet Training script')
   cmd:text('See https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md for examples')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',       '',         'Path to dataset')
   cmd:option('-dataset',    'imagenet', 'Options: imagenet | cifar10 | cifar100')
   cmd:option('-manualSeed', 0,          'Manually set RNG seed')
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
   cmd:option('-iGPU',       1,          'Select a GPU index')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-gen',        'gen',      'Path to save generated files')
   cmd:option('-precision', 'single',    'Options: single | double | half')
   ------------- Data options ------------------------
   cmd:option('-nThreads',        2, 'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',         0,       'Number of total epochs to run')
   cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       32,      'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',        'false', 'Run on validation set only')
   cmd:option('-tenCrop',         'false', 'Ten-crop testing')
   cmd:option('-randCrop',        'true',  'Random crop')
   ------------- Feature Extraction options ----------
   cmd:option('-feOnly',          'false', 'Feature extraction only')
   cmd:option('-feDir',           'feat',  'Directory in which to save features')
   cmd:option('-feLayerIndex',    -1,      'Layer index')
   ------------- Checkpointing options ---------------
   cmd:option('-save',            'checkpoints', 'Directory in which to save checkpoints')
   cmd:option('-resume',          'none',        'Resume from the latest checkpoint in this directory')
   cmd:option('-saveNresume',     'none',        'Save and resume from the checkpoint in this directory')
   ---------- Optimization options ----------------------
   cmd:option('-LR',              0.1,     'initial learning rate')
   cmd:option('-momentum',        0.9,     'momentum')
   cmd:option('-weightDecay',     1e-4,    'weight decay')
   cmd:option('-lsuv',            'false', 'apply layer-sequential unit-variance (LSUV) initialization ')
   ---------- Model options ----------------------------------
   cmd:option('-netType',      'resnet', 'Options: resnet | preresnet')
   cmd:option('-preModel',     'none',   'Path to pretrained Model')
   cmd:option('-preModelAct',  'none',   'Pretrained model activation type')
   cmd:option('-preTarget',    'conv',   'Training target when pretrained model is given (conv | class)')
   cmd:option('-donModel',     'none',   'none | addFC | Path to doner model')
   cmd:option('-chSelector',   'none',   'Path to channel selector')
   cmd:option('-nLastLayerCh', 132,      'Number of last conv layer channels')
   cmd:option('-depth',        34,       'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
   cmd:option('-shortcutType', '',       'Options: A | B | C')
   cmd:option('-wideFactor',   10,       'Network width for wide resnet')
   cmd:option('-dropout',      0.2,      'Dropout for wide resnet')
   cmd:option('-retrain',      'none',   'Path to model to retrain with')
   cmd:option('-retrainOnlyFC','false',  'Retrain last full connection layer only')
   cmd:option('-optimState',   'none',   'Path to an optimState to reload from')
   cmd:option('-criterion',    'none',   'Training criterion')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
   cmd:option('-optnet',          'false', 'Use optnet to reduce memory usage')
   cmd:option('-resetClassifier', 'false', 'Reset the fully connected layer for fine-tuning')
   cmd:option('-nClasses',         10,      'Number of classes in the dataset')
   cmd:text()
     
   local opt = cmd:parse(arg or {})
   
   opt.testOnly = opt.testOnly ~= 'false'
   opt.feOnly = opt.feOnly ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'
   opt.resetClassifier = opt.resetClassifier ~= 'false'
   opt.retrainOnlyFC = opt.retrainOnlyFC ~= 'false'
   opt.randCrop = opt.randCrop ~= 'false'
   opt.lsuv = opt.lsuv ~= 'false'
   
   if opt.saveNresume ~= 'none' then
      opt.save = opt.saveNresume
      opt.resume = opt.saveNresume
   end
    
   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end
   if opt.feOnly then
      if not paths.dirp(opt.feDir) and not paths.mkdir(opt.feDir) then
         cmd:error('error: unable to create feature directory: ' .. opt.feDir .. '\n')
      end
   end
   
   if not opt.testOnly and not opt.feOnly then 
      local fd = io.open(('%s/log_%d.txt'):format(opt.save,math.ceil(sys.clock())), 'w')
      fd:write(table.concat(arg, ' '))
      fd:close()
   end

   if opt.dataset == 'imagenet' then
      -- Handle the most common case of missing -data flag
      local trainDir = paths.concat(opt.data, 'train')
      if not paths.dirp(opt.data) then
         cmd:error('error: missing ImageNet data directory')
      elseif not paths.dirp(trainDir) then
         cmd:error('error: ImageNet missing `train` directory: ' .. trainDir)
      end
      -- Default shortcutType=B and nEpochs=90
      opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs
   elseif opt.dataset == 'cifar10' then
      -- Default shortcutType=A and nEpochs=164
      opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
   elseif opt.dataset == 'cifar100' then
       -- Default shortcutType=A and nEpochs=164
       opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
       opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.precision == nil or opt.precision == 'single' then
      opt.tensorType = 'torch.CudaTensor'
   elseif opt.precision == 'double' then
      opt.tensorType = 'torch.CudaDoubleTensor'
   elseif opt.precision == 'half' then
      opt.tensorType = 'torch.CudaHalfTensor'
   else
      cmd:error('unknown precision: ' .. opt.precision)
   end

   if opt.resetClassifier then
      if opt.nClasses == 0 then
         cmd:error('-nClasses required when resetClassifier is set')
      end
   end
   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   return opt
end

return M
