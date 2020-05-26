import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log10
from torch.distributions.normal import Normal
from optimizers import Lookahead, Lamb # Local file.


'''
Discriminator.
'''

class Discriminator(nn.Module):

   def __init__(self):
      super().__init__()

      self.layers = nn.Sequential(
         # First hidden layer.
         nn.Linear(120, 64),
         nn.ReLU(),
         nn.Dropout(p=.5),
         nn.BatchNorm1d(64),

         # Second hidden layer.
         nn.Linear(64, 64),
         nn.ReLU(),
         nn.Dropout(p=.5),
         nn.BatchNorm1d(64),

         # Third hidden layer.
         nn.Linear(64, 64),
         nn.ReLU(),
         nn.Dropout(p=.5),
         nn.BatchNorm1d(64),

         # Output.
         nn.Linear(64, 1),
      )

   def forward(self, x):
      return self.layers(x)


'''
Data model.
'''

class qPCRData:

   def __init__(self, path, randomize=False):
      def keep(line):
         items = line.split()
         # Remove negative controls.
         if items[0] == 'A01': return False
         if items[0] == 'B01': return False
         # Remove positive controls.
         if items[0] == 'G12': return False
         if items[0] == 'H12': return False
         return True
      def fmt(line):
         min3log = lambda x: log10(x) if x > .001 else -3.
         # Raw data (delta Rn).
         raw = [float(x) for x in line.split()[1:]]
         y = float(raw[44]) + float(raw[89])
         diff = [raw[i+1]-raw[i] for i in range(len(raw)-1)]
         logx = [min3log(x) for x in raw]
         return [y] + diff[-30:] + logx[-45:] + raw[-45:]
      with open(path) as f:
         self.data = [fmt(line) for line in f if keep(line)]

   def batches(self, randomize=False, btchsz=32):
      data = self.data
      # Produce batches in index format (i.e. not text).
      idx = np.arange(len(data))
      if randomize: np.random.shuffle(idx)
      if btchsz > len(idx): btchsz = len(idx)
      # Define a generator for convenience.
      for ix in np.array_split(idx, len(idx) // btchsz):
         y = torch.tensor([data[i][0] for i in ix])
         curves = torch.tensor([data[i][1:] for i in ix])
         yield y, curves


if __name__ == "__main__":

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   disc = Discriminator()

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda':
      disc.cuda()

   if len(sys.argv) > 1:
      data = qPCRData(sys.argv[1], randomize=False)
      disc.load_state_dict(torch.load('crv-pred-500.tch'))
      for y, crv in data.batches(btchsz=4096):
         crv = crv.to(device)
         with torch.no_grad():
            out = disc(crv)
         x = torch.cat([y.view(-1,1), out, crv], 1)
         np.savetxt(sys.stdout, x.cpu().numpy(), fmt='%.4f')
      sys.exit(0)

   train = qPCRData('first.txt')
   test = qPCRData('second.txt')

   lr = 0.0001 # The celebrated learning rate

   dopt = torch.optim.Adam(disc.parameters(), lr=lr)

   # (Binary) cross-entropy loss.
   loss_clsf = nn.MSELoss(reduction='mean')

   for epoch in range(500):
      n_train_batches = 0
      dloss = 0.
      disc.train()
      for y, crv in train.batches(btchsz=92):
         n_train_batches += 1
         y = y.to(device)
         crv = crv.to(device)

         noise = torch.randn(crv.shape) * .1
         disc_loss = loss_clsf(disc(crv + noise).squeeze(), y)
         dloss += float(disc_loss)

         dopt.zero_grad()
         disc_loss.backward()
         dopt.step()


      n_test_batches = 0
      tloss = 0.
      disc.eval()
      for y, crv in test.batches(btchsz=92):
         n_test_batches += 1
         y = y.to(device)
         crv = crv.to(device)

         with torch.no_grad():
            disc_loss = loss_clsf(disc(crv).squeeze(), y)
         tloss += float(disc_loss)


      # Display update at the end of every epoch.
      sys.stderr.write('Epoch %d, disc: %f, test: %f\n' % \
         (epoch+1, dloss / n_train_batches, tloss / n_test_batches))

   # Done, save the networks.
   torch.save(disc.state_dict(), 'crv-pred-%d.tch' % (epoch+1))
