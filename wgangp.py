'''
Wasserstein GANs references:
https://arxiv.org/abs/1701.07875.pdf
https://arxiv.org/pdf/1704.00028.pdf
https://arxiv.org/pdf/1709.08894.pdf
'''

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from optimizers import Lookahead # Local file.

ZDIM = 5

'''
Generator
'''

class Generator(nn.Module):

   def __init__(self):
      super().__init__()

      # The 'ref' parameter will allow seamless random
      # generation on CUDA. It indirectly stores the
      # shape of 'z' but is never updated during learning.
      self.ref = nn.Parameter(torch.zeros(ZDIM))

      self.hidden_layers = nn.Sequential(
         # First hidden layer.
         nn.Linear(ZDIM, 32),
         nn.ReLU(),
         nn.LayerNorm(32),

         # Second hidden layer.
         nn.Linear(32, 64),
         nn.ReLU(),
         nn.LayerNorm(64),

         # Thid hidden layer.
         nn.Linear(64, 128),
         nn.ReLU(),
         nn.LayerNorm(128),
      )

      # The visible layer is a Beta variate.
      self.mu = nn.Linear(128, 135)
      self.sd = nn.Linear(128, 135)

   def detfwd(self, z):
      '''Deterministic part of the generator.'''
      # Transform by passing through the layers.
      h = self.hidden_layers(z)
      # Get the Gaussian parameters.
      mu = self.mu(h)
      sd = F.softplus(self.sd(h))
      return mu, sd

   def forward(self, nsmpl):
      zero = torch.zeros_like(self.ref) # Proper device.
      one = torch.ones_like(self.ref)   # Proper device.
      z = Normal(zero, one).sample([nsmpl])
      a,b = self.detfwd(z)
      return Normal(a,b).rsample()


'''
Discriminator.
'''

class Discriminator(nn.Module):

   def __init__(self):
      super().__init__()

      self.layers = nn.Sequential(
         # First hidden layer.
         nn.Linear(135, 128),
         nn.ReLU(),

         # Second hidden layer.
         nn.Linear(128, 128),
         nn.ReLU(),

         # Third hidden layer.
         nn.Linear(128, 64),
         nn.ReLU(),

         # Visible layer.
         nn.Linear(64, 1),
      )

   def forward(self, x):
      return self.layers(x)


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
         # Raw data (delta Rn).
         raw = [float(x) for x in line.split()[1:]]
         # Take the diff so that numbers are close to 0.
         return [raw[0]] + [raw[i+1]-raw[i] for i in range(len(raw)-1)]
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

   genr = Generator()
   disc = Discriminator()

   genr.load_state_dict(torch.load('genr-wgangp-1000.tch'))
   disc.load_state_dict(torch.load('disc-wgangp-1000.tch'))

   data = qPCRData('qPCR_data.txt', test=False)

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda':
      genr.cuda()
      disc.cuda()
   
   lr = .0001 # The celebrated learning rate
   # Optimizer of the generator (Adam)
   gbase = torch.optim.Adam(genr.parameters(), lr=lr)
   gopt = Lookahead(base_optimizer=gbase, k=5, alpha=0.8)
   # Optimizer of the discriminator (adam)
   dopt = torch.optim.Adam(disc.parameters(), lr=lr)

   for epoch in range(1000):

      wdist = 0.
      batch_is_over = False
      batches = data.batches()

      while True:

         # PHASE I: compute Wasserstein distance.
         for _ in range(5):
            try:
               batch = next(batches)
            except StopIteration:
               batch_is_over = True
               break
            nsmpl = batch.shape[0]
            # Clamp to prevent NaNs in the log-likelihood.
            #real = torch.clamp(batch, min=.01, max=.99).to(device)
            real = batch.to(device)
            with torch.no_grad():
               fake = genr(nsmpl)

            np.savetxt(sys.stdout, fake.cpu().numpy(), fmt='%.4f')
            sys.exit()

            # Compute gradient penalty.
            t = torch.rand(nsmpl, device=device).view(-1,1)
            x = torch.autograd.Variable(t * real + (1-t) * fake,
                  requires_grad=True)
            px = disc(x)
            grad, = torch.autograd.grad(outputs=px, inputs=x,
                  grad_outputs=torch.ones_like(px),
                  create_graph=True, retain_graph=True)
            pen = 10 * torch.relu(grad.norm(2, dim=1) - 1)**2

            # Compute loss and update.
            loss = disc(fake).mean() - disc(real).mean() + pen.mean()

            dopt.zero_grad()
            loss.backward()
            dopt.step()

         # This is the Wasserstein distance.
         wdist += float(-loss)

         # PHASE II: update the generator
         loss = - disc(genr(nsmpl)).mean()

         gopt.zero_grad()
         loss.backward()
         gopt.step()

         if batch_is_over: break

      # Display update at the end of every epoch.
      sys.stderr.write('Epoch %d, wdist: %f\n' % (epoch+1, wdist))

      if (epoch + 1) % 100 == 0:
         # Save the networks.
         torch.save(genr.state_dict(), 'genr-wgangp-%d.tch' % (epoch+1))
         torch.save(disc.state_dict(), 'disc-wgangp-%d.tch' % (epoch+1))
