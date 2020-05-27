#!/usr/bin/env python

import gzip
import os
import sys

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data

from torch.distributions.negative_binomial import NegativeBinomial

WSZ = 100000

class HiCData(torch.utils.data.Dataset):

   def __init__(self, path, sz=0, device='auto'):
      super(HiCData, self).__init__()
      # Use graphics card whenever possible.
      self.device = device if device != 'auto' else \
         'cuda' if torch.cuda.is_available() else 'cpu'
      if sz > 0:
         # The final size 'sz' is specified.
         self.sz = sz
         self.data = torch.zeros((self.sz,self.sz), device=self.device)
         self.add(path)
      else:
         # Store contacts in set 'S' to determine
         # the size 'sz' of the Hi-C contact matrix.
         S = set()
         with gzip.open(path) as f:
            for line in f:
               (a,b,c) = (int(_) for _ in line.split())
               S.add((a,b,c))
               if a > sz: sz = a
               if b > sz: sz = b
         # Write Hi-C matrix as 'pytorch' tensor.
         self.sz = sz / WSZ
         self.data = torch.zeros((self.sz,self.sz), device=self.device)
         for (a,b,c) in S:
            A = a/WSZ-1
            B = b/WSZ-1
            self.data[A,B] = c
            self.data[A,B] = c

   def add(self, path):
      with gzip.open(path) as f:
         for line in f:
            (a,b,c) = (int(_) for _ in line.split())
            if (a/WSZ,b/WSZ) <= self.data.shape:
               A = a/WSZ-1
               B = b/WSZ-1
               self.data[A,B] += c
               self.data[A,B] += c


class Model(nn.Module):

   def __init__(self, HiC):
      super(Model, self).__init__()
      # Data.
      self.HiC = HiC
      self.sz = HiC.sz
      self.device = HiC.device
      # Parameters.
      self.p = nn.Parameter(torch.ones(self.sz, device=self.device))
      self.b = nn.Parameter(torch.ones(1, device=self.device))
      self.a = nn.Parameter(torch.ones(1, device=self.device))
      self.t = nn.Parameter(torch.ones(1, device=self.device))
      self.Z = nn.Parameter(torch.tensor(1e-6, device=self.device))
      self.C = nn.Parameter(torch.tensor(self.sz/3., device=self.device))


   def optimize(self):
      # Mask the diagonal (dominant outlier) and half of
      # the matrix to not double-count the evidence.
      mask = torch.ones((self.sz,self.sz), device=self.device).triu(1)
      # Index vector.
      idx = torch.arange(float(self.sz), device=self.device)
      # Compute log-distances between loci.
      u = torch.ones((self.sz,self.sz), device=self.device).triu()
      dmat = torch.matrix_power(u, 2)
      dmat = dmat + torch.t(dmat)
      dmat[dmat == 0] = 1.0
      dmat = torch.log(dmat)
      # Optimizer and scheduler.
      optim = torch.optim.Adam(self.parameters(), lr=.01)
      sched = torch.optim.lr_scheduler.MultiStepLR(optim,
            milestones=[3000])
      # Weights (mappability biases etc.)
      rowsums = torch.sum(self.HiC.data, 1)
      remove = torch.diag(self.HiC.data) / rowsums > .25
      w = torch.sqrt(rowsums - torch.diag(self.HiC.data))
      w[remove] = 0.0
      W = torch.ger(w,w)
      # Gradient descent proper.
      import pdb; pdb.set_trace()
      for step in range(3200):
         # Take the sigmoid to constrain 'P' within (0,1).
         P = torch.sigmoid(self.p)
         x = P*self.b - (1-P)*self.b
         AB = torch.ger(x,x)
         # Megadomains: use a sharp logistic drop (decay 3).
         V = 1. / (1. + torch.exp(-5*(idx - self.C)))
         #M = self.Z * (torch.ger(V,V) + torch.ger(1-V,1-V))
         M = self.Z * torch.ger(V,V)
         # Expected counts.
         mu = W * torch.exp(AB - dmat*self.a + M)
         # Reparametrize for the negative binomial.
         nbp = mu / (mu + self.t)
         log_p = NegativeBinomial(self.t, nbp).log_prob(self.HiC.data)
         # Multiply by mask to remove entries.
         ll = -torch.sum(log_p * mask)
         optim.zero_grad()
         ll.backward()
         optim.step()
         sched.step()
         #sys.stderr.write('%d %f\n' % (step, float(ll)))
         
      sys.stdout.write('# alpha %f\n' % float(self.a))
      sys.stdout.write('# beta %f\n' % float(self.b))
      sys.stdout.write('# theta %f\n' % float(self.t))
      for i in range(self.sz):
         x = float(torch.sigmoid(self.p[i]))
         if abs(x - 0.73105857) > 1e-5:
            sys.stdout.write('%d\t%f\n' % (i, x))
         else:
            sys.stdout.write('%d\tNA\n' % i)

if __name__ == '__main__':
   sz = int(sys.argv.pop(1)) if sys.argv[1].isdigit() else 0
   sys.stderr.write('%s\n' % sys.argv[1])
   HiC = HiCData(path=sys.argv[1], sz=sz)
   for fname in sys.argv[2:]:
      sys.stderr.write('%s\n' % fname)
      HiC.add(fname)
   M = Model(HiC)
   M.optimize()
