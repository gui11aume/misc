import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from optimizers import Lookahead, Lamb # Local file.

from qPCR import qPCRData

ZDIM = 3

'''
Encoder.
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
         nn.Linear(45, 64),
         nn.ReLU(),
         nn.Dropout(p=0.1),
         nn.LayerNorm(64),

         # Second hidden layer.
         nn.Linear(64, 32),
         nn.ReLU(),
         nn.Dropout(p=0.1),
         nn.LayerNorm(32),

         # Third hidden layer.
         nn.Linear(32, 16),
         nn.ReLU(),
         nn.Dropout(p=0.1),
         nn.LayerNorm(16),

         # Welcome to the latent space.
         nn.Linear(16, ZDIM),
      )

   def forward(self, x):
      return self.hidden_layers(x)


'''
Decoder.
'''

class Decoder(nn.Module):

   def __init__(self):
      super().__init__()

      # The 'ref' parameter will allow seamless random
      # generation on CUDA. It indirectly stores the
      # shape of 'z' but is never updated during learning.
      self.ref = nn.Parameter(torch.zeros(ZDIM))

      # Three hidden layers.
      self.hidden_layers = nn.Sequential(
         # First hidden layer.
         nn.Linear(ZDIM, 16),
         nn.ReLU(),
         nn.LayerNorm(16),

         # Second hidden layer.
         nn.Linear(16, 32),
         nn.ReLU(),
         nn.LayerNorm(32),

         # Third hidden layer.
         nn.Linear(32, 64),
         nn.ReLU(),
         nn.LayerNorm(64),
      )

      self.mu = nn.Linear(64, 45)
      self.sd = nn.Linear(64, 45)

   def forward(self, x):
      h = self.hidden_layers(x)
      return self.mu(h), F.softplus(self.sd(h))


if __name__ == "__main__":

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   encd = Encoder()
   decd = Decoder()

   train = qPCRData('first.txt', create_test=False, randomize=True)
   test = qPCRData('second.txt', create_test=False, randomize=True)

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda':
      encd.cuda()
      decd.cuda()

   if len(sys.argv) > 1:
      encd.load_state_dict(torch.load('gaae-encd-200.tch'))
      decd.load_state_dict(torch.load('gaae-decd-200.tch'))
      data = qPCRData(sys.argv[1], create_test=False, randomize=False)
      for batch in data.batches(use_test=False, randomize=False):
         batch = batch[:,:45].to(device)
         with torch.no_grad():
            mu, sd = decd(encd(batch))
            y = Normal(mu, sd).sample()
         out = torch.cat([batch,y], dim=-1)
         out = torch.cat([batch,mu], dim=-1)
         np.savetxt(sys.stdout, out.cpu().numpy(), fmt='%.4f')
      sys.exit(0)

   lr = 0.001 # The learning rate

   aopt = torch.optim.Adam(encd.parameters(), lr=lr)
   bopt = torch.optim.Adam(decd.parameters(), lr=lr)

   asched = torch.optim.lr_scheduler.MultiStepLR(aopt, [1000])
   bsched = torch.optim.lr_scheduler.MultiStepLR(bopt, [1000])

   def compute_batch_loss(batch, device):
      batch = batch[:,:45].to(device) # N1
      nx = batch.numel()

      #noise = torch.randn(batch.shape, device=device) * .2
      mu, sd = decd(encd(batch))

      # Gaussian loss.
      return -Normal(mu, sd).log_prob(batch).sum() / nx

   for epoch in range(200):
      n_train_btch = n_test_btch = 0
      train_loss = test_loss = 0.

      # Training phase.
      encd.train()
      decd.train()
      for batch in train.batches(btchsz=128):
         n_train_btch += 1

         loss = compute_batch_loss(batch, device)
         train_loss += float(loss)

         aopt.zero_grad()
         bopt.zero_grad()
         loss.backward()
         aopt.step()
         bopt.step()

      # Testing phase.
      encd.eval()
      decd.eval()
      for batch in test.batches(btchsz=2048):
         n_test_btch += 1

         loss = compute_batch_loss(batch, device)
         test_loss += float(loss)

      # Update schedulers.
      asched.step()
      bsched.step()

      # Display update at the end of every epoch.
      sys.stderr.write('Epoch %d, train: %f, test: %f\n' % \
         (epoch+1, train_loss / n_train_btch, test_loss / n_test_btch))

   # Done, save the networks.
   torch.save(encd.state_dict(), 'gaae-encd-%d.tch' % (epoch+1))
   torch.save(decd.state_dict(), 'gaae-decd-%d.tch' % (epoch+1))
