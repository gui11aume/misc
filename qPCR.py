import numpy as np
import torch

'''
Data model.
'''

class qPCRData:

   def __init__(self, path, randomize=False, create_test=False):
      with open(path) as f:
         self.data = [self.fmt(line) for line in f if self.keep(line)]
      # Create train and test data.
      if create_test:
         if randomize: np.random.shuffle(self.data)
         sztest = len(self.data) // 10 # 10% for the test.
         self.test = self.data[-sztest:]
         self.data = self.data[:-sztest]

   def fmt(self, line, diff=False):
      # Raw data (delta Rn).
      raw = [float(x) for x in line.split()[1:]]
      if diff:
         # Take the diff so that numbers are close to 0.
         diffN1 = [raw[0]]  + [raw[i+1]-raw[i] for i in range(0,44)]
         diffN2 = [raw[45]] + [raw[i+1]-raw[i] for i in range(45,89)]
         diffRP = [raw[90]] + [raw[i+1]-raw[i] for i in range(90,134)]
         return diffN1 + diffN2 + diffRP
      else:
         return raw

   def keep(self, line):
      items = line.split()
      # Remove negative controls.
      if items[0] == 'A01': return False
      if items[0] == 'B01': return False
      # Remove positive controls.
      if items[0] == 'G12': return False
      if items[0] == 'H12': return False
      return True

   def batches(self, use_test=False, randomize=True, btchsz=32):
      data = self.test if use_test else self.data
      # Produce batches in index format (i.e. not text).
      idx = np.arange(len(data))
      if randomize: np.random.shuffle(idx)
      if btchsz > len(idx): btchsz = len(idx)
      # Define a generator for convenience.
      for ix in np.array_split(idx, len(idx) // btchsz):
         yield torch.tensor([data[i] for i in ix])
