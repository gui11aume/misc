import blitz
import blitz.modules
import blitz.utils
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Discriminator.
'''

@blitz.utils.variational_estimator
class Discriminator(nn.Module):

   def __init__(self):
      super().__init__()

      self.layers = nn.Sequential(
         # First (Bayesian) hidden layer.
         blitz.modules.BayesianLinear(44, 16),
         nn.ReLU(),
         nn.BatchNorm1d(16),

         # Second (Bayesian) hidden layer.
         blitz.modules.BayesianLinear(16, 16),
         nn.ReLU(),
         nn.BatchNorm1d(16),

         # Output in linear space (classification).
         blitz.modules.BayesianLinear(16, 1)
      )

   def forward(self, x):
      return self.layers(x)


'''
Parser.
'''

def parse_clipped(fname):
   # Parse exported 7900HT "clipped" file.
   RP = dict()
   with open(fname) as f:
      _ = next(f) # SDS version.
      _ = next(f) # Column names.
      for line in f:
         items = line.split()
         well = int(items[0])
         dRn = [float(x) for x in items[48:]]
         # Translate 384-well ID into 96-well ID.
         row = (well - 1) // 24
         col = (well - 1) % 24
         # RP is exclusievely in even row and odd column.
         # But Pyton is 0-based, so...
         if (row % 2 != 1) or (col % 2 != 0):
            continue
         code = 'ABCDEFGH'[row//2] + ('%02d' % (1+col//2))
         RP[code] = dRn
   # Remove positive controls (but keep positives).
   out = frozenset(['G12', 'H12'])
   return torch.tensor([RP[k] for k in sorted(RP) if k not in out])


if __name__ == "__main__":

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   disc = Discriminator()

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda':
      disc.cuda()

   disc.load_state_dict(torch.load('crv-disc-100.tch'))

   data = parse_clipped(sys.argv[1]).to(device)
   with torch.no_grad():
      # Probability of a positive (for N1).
      ppos = [1. - torch.sigmoid(disc(data)) for _ in range(100)]
   median = torch.median(torch.cat(ppos, dim=-1), dim=-1)
   # A median less than 0.003 means very little chance that N1 is
   # positive. According to the calibration, it is something like
   # 0.5% for an average around 8% positives for N1.
   bad = median.values < 0.003
   print(bad)
   # IMPORTANT: I left the positive controls on the way. You should
   # add back your own code and put an arbitrary cut-off somewhere.
   # It is not relevant to test them with this classifier because
   # positive controls are an artificial mix that obeys different
   # rules.
