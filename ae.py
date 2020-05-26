import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from optimizers import Lookahead # Local file.

ZDIM = 7

'''
Encoder-Decoder
'''

class Encoder(nn.Module):

   def __init__(self):
      super().__init__()

      # The 'ref' parameter will allow seamless random
      # generation on CUDA. It indirectly stores the
      # shape of 'z' but is never updated during learning.
      self.ref = nn.Parameter(torch.zeros(ZDIM))

      # Three hidden layers.
      self.hidden_layers = nn.Sequential(
         # First hidden layer.
         nn.Linear(90, 64),
         nn.LayerNorm(64),
         nn.ReLU(),

         # Second hidden layer.
         nn.Linear(64, 32),
         nn.LayerNorm(32),
         nn.ReLU(),

         # Third hidden layer.
         nn.Linear(32, 16),
         nn.LayerNorm(16),
         nn.ReLU(),

         # Welcome to the latent space.
         nn.Linear(16, ZDIM),
      )

   def rnd(self, nsmpl):
      one = torch.ones_like(self.ref)  # On the proper device.
      return Normal(0. * one, one).sample([nsmpl])

   def forward(self, x):
      return self.hidden_layers(x)


class Decoder(nn.Module):

   def __init__(self):
      super().__init__()

      # The 'ref' parameter will allow seamless random
      # generation on CUDA. It indirectly stores the
      # shape of 'z' but is never updated during learning.
      self.ref = nn.Parameter(torch.zeros(2))

      # Two hidden layers.
      self.hidden_layers = nn.Sequential(
         # First hidden layer.
         nn.Linear(ZDIM, 16),
         nn.LayerNorm(16),
         nn.ReLU(),

         # Second hidden layer.
         nn.Linear(16, 32),
         nn.LayerNorm(32),
         nn.ReLU(),

         # Third hidden layer.
         nn.Linear(32, 64),
         nn.LayerNorm(64),
         nn.ReLU(),
      )

      # Visible layer.
      self.last = nn.Linear(64, 90)

   def forward(self, z):
      # Transform by passing through the layers.
      h = self.hidden_layers(z)
      return self.last(h)


'''
Data model.
'''

class qPCRData:

   def __init__(self, path, randomize=True, test=True):
      def keep(line):
         # Remove negative controls.
         if line.startswith('A1'): return False
         if line.startswith('B1'): return False
         # Remove positive controls.
         if line.startswith('G12'): return False
         if line.startswith('H12'): return False
         return True
      def fmt(line):
         return [float(x) for x in line.split()[1:91]]
      with open(path) as f:
         self.data = [fmt(line) for line in f if keep(line)]
      # Create train and test data.
      if test:
         if randomize: np.random.shuffle(self.data)
         sztest = len(self.data) // 10 # 10% for the test.
         self.test = self.data[-sztest:]
         self.data = self.data[:-sztest]

   def batches(self, test=False, randomize=True, btchsz=32):
      data = self.test if test else self.data
      # Produce batches in index format (i.e. not text).
      idx = np.arange(len(data))
      if randomize: np.random.shuffle(idx)
      # Define a generator for convenience.
      for ix in np.array_split(idx, len(idx) // btchsz):
         yield torch.tensor([data[i] for i in ix])


if __name__ == "__main__":

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   encd = Encoder()
   decd = Decoder()

#   encd.load_state_dict(torch.load('encd-200.tch'))
#   decd.load_state_dict(torch.load('decd-200.tch'))
#   disc.load_state_dict(torch.load('disc-200.tch'))

   data = qPCRData('qPCR_data.txt', test=False)

   with torch.cuda.device(1):
      # Do it with CUDA if possible.
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      if device == 'cuda':
         encd.cuda()
         decd.cuda()

   #   for batch in data.batches():
   #      batch = batch.to(device)
   #      with torch.no_grad():
   #         z = encd(batch)
   #         y = decd(z)
   #      np.savetxt(sys.stdout, torch.cat([batch,y], dim=1).cpu().numpy(),
   #            fmt='%.4f')
   #   sys.exit()
      
      lr = 0.001 # The celebrated learning rate

      # Optimizer of the encoder (Adam).
      abase = torch.optim.Adam(encd.parameters(), lr=lr)
      aopt = Lookahead(base_optimizer=abase, k=5, alpha=0.8)
      # Optimizer of the decoder (Adam).
      bbase = torch.optim.Adam(decd.parameters(), lr=lr)
      bopt = Lookahead(base_optimizer=bbase, k=5, alpha=0.8)

      # Mean square error losss.
      loss_cstr = nn.MSELoss(reduction='mean')

      for epoch in range(200):
         closs = 0.
         for batch in data.batches():
            nsmpl = batch.shape[0]
            batch = batch.to(device)

            one = torch.ones(1).to(device)
            noise = Normal(0 * one, .2 * one).sample(batch.shape)
            z = encd(batch + noise.view(batch.shape)) # Add Gaussian noise.
            y = decd(z)

            loss = loss_cstr(batch, y)
            closs += float(loss)

            aopt.zero_grad()
            bopt.zero_grad()
            loss.backward()
            aopt.step()
            bopt.step()

         # Display update at the end of every epoch.
         sys.stderr.write('Epoch %d, cstr: %f\n' % \
               (epoch+1, closs))

      # Done, save the networks.
      torch.save(encd.state_dict(), 'encd-%d.tch' % (epoch+1))
      torch.save(decd.state_dict(), 'decd-%d.tch' % (epoch+1))

      for batch in data.batches():
         batch = batch.to(device)
         with torch.no_grad():
            z = encd(batch)
            y = decd(z)
         np.savetxt(sys.stdout, torch.cat([batch,y], dim=1).cpu().numpy(),
               fmt='%.4f')
