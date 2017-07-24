--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  CIFAR-10 dataset loader
--

local t = require 'datasets/transforms'

local M = {}
local SVHNDataset = torch.class('resnet.SVHNDataset', M)

function SVHNDataset:__init(imageInfo, opt, split)
   assert(imageInfo[split], split)
   self.imageInfo = imageInfo[split]
   self.split = split
end

function SVHNDataset:get(i)
   local image = self.imageInfo.data[i]:float()
   local label = self.imageInfo.labels[i]

   return {
      input = image,
      target = label,
   }
end

function SVHNDataset:size()
   return self.imageInfo.data:size(1)
end

-- Computed from entire CIFAR-10 training set
--local meanstd = {
--   mean = {125.3, 123.0, 113.9},
--   std  = {63.0,  62.1,  66.7},
--}

function SVHNDataset:preprocess(isRandCrop)
   if self.split == 'train' then
      if isRandCrop then 
         return t.Compose{
--            t.ColorNormalize(meanstd),
            t.HorizontalFlip(0.5),
            t.RandomCrop(32, 4),
          }
      else
         return t.HorizontalFlip(0)
      end
   elseif self.split == 'val' then
      return t.HorizontalFlip(0)
   else
      error('invalid split: ' .. self.split)
   end
end

return M.SVHNDataset
