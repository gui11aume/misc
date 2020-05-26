import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from optimizers import Lamb, Lookahead # Local file.

'''
Dot product attention layer.
'''

class Attention(nn.Module):

   def init_matrix(self, *dims):
      m = torch.Tensor(*dims)
      # Taken from the source code of 'torch.nn.Linear'.
      torch.nn.init.kaiming_uniform_(m, a=np.sqrt(5))
      return m

   def __init__(self, h, d_model, dropout=0.1):
      assert d_model % h == 0 # Just to be sure.
      super().__init__()
      self.h = h
      self.d = d_model

      # Linear transformations of embeddings.
      self.Wq = nn.Parameter(self.init_matrix(d_model, d_model))
      self.Wk = nn.Parameter(self.init_matrix(d_model, d_model))
      self.Wv = nn.Parameter(self.init_matrix(d_model, d_model))

      # Content biases.
      self.cb = nn.Parameter(torch.zeros(d_model)) # Content bias.

      # Output layers.
      self.do = nn.Dropout(p=dropout)
      self.Wo = nn.Linear(d_model, d_model)
      self.ln = nn.LayerNorm(d_model)

   def forward(self, X, Y):
      '''
            X  ~  (Batch, L, d_model)
           W.  ~  (d_model, d_model)
           cb  ~  (1, h, 1, d_model/h)
        q,k,v  ~  (Batch, h, L, d_model/h)
            A  ~  (Batch, h, L, L)
           Oh  ~  (Batch, h, d_model/h, L)
            O  ~  (Batch, L, d_model)
      '''

      h  = self.h       # Number of heads.
      H  = self.d // h  # Head dimension.
      N  = X.shape[0]   # Batch size.
      L1 = X.shape[1]   # Text length (X).
      L2 = Y.shape[1]   # Text length (Y).

      # This model is only going to work if L1 is the same as L2.
      assert L1 == L2

      # Linear transforms.
      q = torch.matmul(X, self.Wq).view(N,L1,h,-1).transpose(1,2)
      k = torch.matmul(X, self.Wk).view(N,L2,h,-1).transpose(1,2)
      v = torch.matmul(X, self.Wv).view(N,L2,h,-1).transpose(1,2)

      # Reshapes.
      cb = self.cb.view(1,h,1,-1).repeat(N,1,L1,1)

      # Dot products.
      A_a = torch.matmul(q,  k.transpose(-2,-1))
      A_c = torch.matmul(cb, k.transpose(-2,-1))

      # Raw attention matrix.
      A = A_a + A_c

      # Attention softmax.
      p_attn = F.softmax(A, dim=-1)

      # Apply attention to v.
      Oh = torch.matmul(p_attn, v)

      # Concatenate attention output.
      O = Oh.transpose(1,2).contiguous().view_as(X)

      # Layer norm and residual connection.
      return self.ln(X + self.do(self.Wo(O)))


'''
Feed foward layer.
'''

class FeedForwardNet(nn.Module):
   def __init__(self, d_model, d_ffn, dropout=0.1):
      super().__init__()
      self.ff = nn.Sequential(
         nn.Linear(d_model, d_ffn),
         nn.ReLU(),
         nn.Linear(d_ffn, d_model)
      )
      self.do = nn.Dropout(p=dropout)
      self.ln = nn.LayerNorm(d_model)

   def forward(self, X):
      return self.ln(X + self.do(self.ff(X)))


''' 
Attention block.
'''

class AttnBlock(nn.Module):
   def __init__(self, h, d_model, d_ffn, dropout=0.1):
      super().__init__()
      self.h = h
      self.d = d_model
      self.f = d_ffn
      self.attn = Attention(h, d_model, dropout=dropout)
      self.ffn  = FeedForwardNet(d_model, d_ffn, dropout=dropout)

   def forward(self, X, Y):
      return self.ffn(self.attn(X, Y))


'''
Classifier.
'''

class Classifier(nn.Module):
   def __init__(self):
      super().__init__()
      self.d_model = 128
      self.d_ffn = 256

      self.embed_context = nn.Sequential(
         nn.Linear(45, self.d_model),
         nn.ReLU(),
         nn.Dropout(p=.3),

         nn.Linear(self.d_model, self.d_model),
         nn.ReLU(),
         nn.Dropout(p=.3),
      )

      self.embed_calls = nn.Linear(44, self.d_model)
      self.attn1 = AttnBlock(4, self.d_model, self.d_ffn)
      self.attn2 = AttnBlock(4, self.d_model, self.d_ffn)
      self.attn3 = AttnBlock(4, self.d_model, self.d_ffn)

      self.final = nn.Linear(self.d_model, 2)

   def forward(self, X, Y):
      # Get context from 'Y'.
      Y = self.embed_context(Y)
      # Make predictions on 'X'.
      X = self.embed_calls(X)
      X = self.attn1(X, Y)
      X = self.attn2(X, Y)
      X = self.attn3(X, Y)
      return self.final(X)


'''
Data model.
'''

class qPCRData:

   def __init__(self, path, randomize=True, test=True):
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
         # Raw data (delta Rn).
         raw = [float(x) for x in line.split()[1:]]
         neg = float(raw[44]) < .25 and float(raw[89]) < .25
         # Take the diff so that numbers are close to 0.
         diff = [raw[i+1]-raw[i] for i in range(len(raw)-1)]
         return [neg] + diff[-44:]
      with open(path) as f:
         self.data = [fmt(line) for line in f if keep(line)]
      # Create train and test data.
      if test:
         if randomize: np.random.shuffle(self.data)
         sztest = len(self.data) // 5 # 20% for the test.
         self.test = self.data[-sztest:]
         self.data = self.data[:-sztest]

   def batches(self, test=False, randomize=False, btchsz=92):
      data = self.test if test else self.data
      # Produce batches in index format (i.e. not text).
      idx = np.arange(len(data))
      if randomize: np.random.shuffle(idx)
      if btchsz > len(idx): btchsz = len(idx)
      # Define a generator for convenience.
      for ix in np.array_split(idx, len(idx) // btchsz):
         neg = torch.LongTensor([data[i][:1] for i in ix])
         curves = torch.tensor([data[i][1:] for i in ix])
         yield neg, curves


if __name__ == "__main__":

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   model = Classifier()

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda': model.cuda()

   if len(sys.argv) > 1:
      model.load_state_dict(torch.load('att-disc-150.tch'))
      test = qPCRData(sys.argv[1], randomize=False, test=False)
      for neg, crv in test.batches(btchsz=92):
         ctx = torch.cat([crv, neg.float()], -1).unsqueeze(0)
         cls = crv.unsqueeze(0)

         ctx = ctx.to(device)
         cls = cls.to(device)
         neg = neg.to(device)

         with torch.no_grad():
            out = F.softmax(model(cls, ctx), dim=-1).squeeze()
         np.savetxt(sys.stdout, out.cpu().numpy(), fmt='%.4f')
      sys.exit(0)


   def compute_batch_loss(neg, crv, device):

      # Cross entropy loss function.
      cost = nn.CrossEntropyLoss(reduction='mean')

      ctx = torch.cat([crv, neg.float()], -1).unsqueeze(0)
      cls = crv.unsqueeze(0)

      ctx = ctx.to(device)
      cls = cls.to(device)
      neg = neg.to(device)

      return cost(model(cls, ctx).squeeze(), neg.squeeze())


   train = qPCRData('first.txt', randomize=False, test=False)
   test = qPCRData('second.txt', randomize=False, test=False)

   lr = 0.0001 # The celebrated learning rate.

   # Optimizer (Lookahead with Lamb).
   baseopt = Lamb(model.parameters(),
         lr=lr, weight_decay=0.01, betas=(.9, .999), adam=True)
   opt = Lookahead(base_optimizer=baseopt, k=5, alpha=0.8)

   for epoch in range(150):
      epoch_loss = 0.
      n_train_btch = 0
      model.train()
      for neg, crv in train.batches(btchsz=92):
         n_train_btch += 1
         loss = compute_batch_loss(neg, crv, device)
         epoch_loss += float(loss)
         # Update.
         opt.zero_grad()
         loss.backward()
         opt.step()

      test_loss = 0.
      n_test_btch = 0
      model.eval()
      for neg, crv in test.batches(btchsz=92):
         n_test_btch += 1
         test_loss += float(compute_batch_loss(neg, crv, device))


      # Display update at the end of every epoch.
      sys.stderr.write('Epoch %d, train: %f, test: %f\n' % \
         (epoch+1, epoch_loss / n_train_btch, test_loss / n_test_btch))

   # Done, save the networks.
   torch.save(model.state_dict(), 'att-disc-%d.tch' % (epoch+1))
