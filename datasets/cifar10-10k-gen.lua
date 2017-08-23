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

local URL = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'

local M = {}
local function convertToTensor(files)
   local data, labels

   for _, file in ipairs(files) do
      local m = torch.load(file, 'ascii')
      if not data then
         data = m.data:t()
         labels = m.labels:squeeze()
      else
         data = torch.cat(data, m.data:t(), 1)
         labels = torch.cat(labels, m.labels:squeeze())
      end
   end

   -- This is *very* important. The downloaded files have labels 0-9, which do
   -- not work with CrossEntropyCriterion
   labels:add(1)

   return {
      data = data:contiguous():view(-1, 3, 32, 32),
      labels = labels,
   }
end

local function convertToTensor2(files)
   local data, labels, data1, labels1, data2, labels2

   for _, file in ipairs(files) do
      local m = torch.load(file, 'ascii')
      if not data then
         data = m.data:t()
         labels = m.labels:squeeze()
      else
         data = torch.cat(data, m.data:t(), 1)
         labels = torch.cat(labels, m.labels:squeeze())
      end
   end

   -- This is *very* important. The downloaded files have labels 0-9, which do
   -- not work with CrossEntropyCriterion
   labels:add(1)
   local yy, ii = torch.sort(labels)

   for i=1,10 do
      if not data1 then
         data1 = data:index(1,ii:narrow(1,1,1000))
         labels1 = labels:index(1,ii:narrow(1,1,1000))
         data2 = data:index(1,ii:narrow(1,1001,4000))
         labels2 = labels:index(1,ii:narrow(1,1001,4000))
      else
         data1 = torch.cat(data1, data:index(1, ii:narrow(1,5000*(i-1)+1,1000)), 1)
         labels1 = torch.cat(labels1, labels:index(1, ii:narrow(1,5000*(i-1)+1,1000)))
         data2 = torch.cat(data2, data:index(1, ii:narrow(1,5000*(i-1)+1001,4000)), 1)
         labels2 = torch.cat(labels2, labels:index(1, ii:narrow(1,5000*(i-1)+1001,4000)))
      end
   end
   local perm = torch.randperm(10000)
   local perm2 = torch.randperm(40000)
--   print(ii[1])
--   print(ii[2])
--   print(ii[3])
--   print(ii[4])
--   print(ii[5])
--   print(ii[6])
--   print(ii[7])
--   print(ii[8])
--   print(ii[9])
--   print(ii[10])
--   prin()
   data = data1:index(1,perm:long())
   labels = labels1:index(1,perm:long())
   data1 = data2:index(1,perm2:long())
   labels1 = labels2:index(1,perm2:long())
   print(labels1)
   print(labels1[torch.eq(labels1,1)]:sum())
   print(labels1[torch.eq(labels1,2)]:sum())
   print(labels1[torch.eq(labels1,3)]:sum())
   print(labels1[torch.eq(labels1,4)]:sum())
   print(labels1[torch.eq(labels1,5)]:sum())
   print(labels1[torch.eq(labels1,6)]:sum())
   print(labels1[torch.eq(labels1,7)]:sum())
   print(labels1[torch.eq(labels1,8)]:sum())
   print(labels1[torch.eq(labels1,9)]:sum())
   print(labels1[torch.eq(labels1,10)]:sum())
print(labels1:size())
print(data1:size())
print(labels:size())
print(data:size())
--require 'image'
--
--local vis = image.toDisplayTensor(data:index(1,torch.range(1,16):long()):view(-1, 3, 32, 32))
--
--image.save('dfafd.png', vis)
--print (labels:index(1,torch.range(1,16):long()))
--   prin()
   return {
      data = data:contiguous():view(-1, 3, 32, 32),
      labels = labels,
   },{
      data = data1:contiguous():view(-1, 3, 32, 32),
      labels = labels1,
   }
end

function M.exec(opt, cacheFile)
   print("=> Downloading CIFAR-10 dataset from " .. URL)
   --local ok = os.execute('curl ' .. URL .. ' | tar xz -C gen/')
   --assert(ok == true or ok == 0, 'error downloading CIFAR-10')

   print(" | combining dataset into a single file")
   local trainData, valiData = convertToTensor2({
      'gen/cifar-10-batches-t7/data_batch_1.t7',
      'gen/cifar-10-batches-t7/data_batch_2.t7',
      'gen/cifar-10-batches-t7/data_batch_3.t7',
      'gen/cifar-10-batches-t7/data_batch_4.t7',
      'gen/cifar-10-batches-t7/data_batch_5.t7',
   })
   local testData = convertToTensor({
      'gen/cifar-10-batches-t7/test_batch.t7',
   })

   print(" | saving CIFAR-10 dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = trainData,
      vali = valiData,
      val = testData,
   })
end

return M
