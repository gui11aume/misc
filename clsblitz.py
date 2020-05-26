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
         # First hidden layer.
         blitz.modules.BayesianLinear(44, 16),
         nn.ReLU(),
         nn.BatchNorm1d(16),

         # Second hidden layer.
         blitz.modules.BayesianLinear(16, 16),
         nn.ReLU(),
         nn.BatchNorm1d(16),

         # Output.
         blitz.modules.BayesianLinear(16, 1)
      )

   def forward(self, x):
      return self.layers(x)


'''
Data model.
'''

class qPCRData:

   def __init__(self, path, randomize=False, test=False):
      def keep(line):
         items = line.split()
         # Remove negative controls.
         #if items[0] == 'A01': return False
         #if items[0] == 'B01': return False
         # Remove positive controls.
         if items[0] == 'G12': return False
         if items[0] == 'H12': return False
         return True
      def fmt(line):
         # Raw data (delta Rn).
         raw = [float(x) for x in line.split()[1:]]
         neg = float(raw[44]) < .25
         # Take the diff so that numbers are close to 0.
         #diff = [raw[i+1]-raw[i] for i in range(len(raw)-1)]
         #return [neg] + diff[-44:]
         return [neg] + raw[-44:]
      with open(path) as f:
         self.data = [fmt(line) for line in f if keep(line)]
      # Create train and test data.
      if test:
         if randomize: np.random.shuffle(self.data)
         sztest = len(self.data) // 5 # 20% for the test.
         self.test = self.data[-sztest:]
         self.data = self.data[:-sztest]

   def batches(self, test=False, randomize=False, btchsz=32):
      data = self.test if test else self.data
      # Produce batches in index format (i.e. not text).
      idx = np.arange(len(data))
      if randomize: np.random.shuffle(idx)
      if btchsz > len(idx): btchsz = len(idx)
      # Define a generator for convenience.
      for ix in np.array_split(idx, len(idx) // btchsz):
         neg = torch.LongTensor([data[i][0] for i in ix])
         curves = torch.tensor([data[i][1:] for i in ix])
         yield neg, curves


if __name__ == "__main__":

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   disc = Discriminator()

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda':
      disc.cuda()

   if len(sys.argv) > 1:
      data = qPCRData(sys.argv[1], test=False, randomize=False)
      disc.load_state_dict(torch.load('crv-disc-100.tch'))
      for neg, crv in data.batches(btchsz=92):
         pos = 1. - neg.to(device)
         crv = crv.to(device)
         with torch.no_grad():
#            out = [F.softmax(disc(crv), 1)[:,:1] for _ in range(100)]
            out = [1. - torch.sigmoid(disc(crv)) for _ in range(100)]
         outcat = torch.cat(out, dim=-1)
         x = torch.cat([pos.view(-1,1), outcat, crv], 1)
         np.savetxt(sys.stdout, x.cpu().numpy(), fmt='%.4f')
      sys.exit(0)

   train = qPCRData('first.txt')
   test = qPCRData('second.txt')

   lr = 0.0001 # The celebrated learning rate

   dopt = torch.optim.Adam(disc.parameters(), lr=lr)

   # (Binary) cross-entropy loss.
   wgt = torch.tensor([10.]).to(device)
   loss_clsf = nn.BCEWithLogitsLoss(pos_weight=wgt, reduction='mean')

   for epoch in range(100):
      n_train_batches = 0
      dloss = 0.
      disc.train()
      for neg, crv in train.batches(btchsz=94):
         n_train_batches += 1
         neg = neg.float().to(device)
         crv = crv.to(device)

         disc_loss = loss_clsf(disc(crv).squeeze(), neg)
         dloss += float(disc_loss)

         dopt.zero_grad()
         disc_loss.backward()
         dopt.step()


      n_test_batches = 0
      tloss = 0.
      disc.eval()
      for neg, crv in test.batches(btchsz=94):
         n_test_batches += 1
         neg = neg.float().to(device)
         crv = crv.to(device)

         with torch.no_grad():
            disc_loss = loss_clsf(disc(crv).squeeze(), neg)
         tloss += float(disc_loss)


      # Display update at the end of every epoch.
      sys.stderr.write('Epoch %d, disc: %f, test: %f\n' % \
         (epoch+1, dloss / n_train_batches, tloss / n_test_batches))

   # Done, save the networks.
   torch.save(disc.state_dict(), 'crv-disc-%d.tch' % (epoch+1))
