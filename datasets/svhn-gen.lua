--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This automatically downloads the CIFAR-10 dataset from
--  http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz
--

local URL = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/svhn.t7.tgz'

local M = {}

local function convertToTensor(files)
   local data, labels

   for _, file in ipairs(files) do
      local m = torch.load(file, 'ascii')
      if not data then
         data = m.X:transpose(3,4)
         labels = m.y[1]:squeeze()
      else
         data = torch.cat(data, m.X:transpose(3,4), 1)
         labels = torch.cat(labels, m.y[1]:squeeze())
      end
   end

   -- This is *very* important. The downloaded files have labels 0-9, which do
   -- not work with CrossEntropyCriterion
--   labels:add(1)
   print(labels)

   return {
      data = data:contiguous(),--:view(-1, 3, 32, 32),
      labels = labels,
   }
end

function M.exec(opt, cacheFile)
   print("=> Downloading SVHN-10 dataset from " .. URL)
   local ok = os.execute('curl ' .. URL .. ' | tar xz -C gen/')
   assert(ok == true or ok == 0, 'error downloading SVHN')

   print(" | combining dataset into a single file")
   local trainData = convertToTensor({
      'gen/housenumbers/train_32x32.t7',
      'gen/housenumbers/extra_32x32.t7',
   })
   local testData = convertToTensor({
      'gen/housenumbers/test_32x32.t7',
   })

   print(" | saving SVHN dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = trainData,
      val = testData,
   })
end

return M
