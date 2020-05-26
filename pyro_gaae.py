import numpy as np
import pyro
import pyro.distributions as dist
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.constraints as constraints

from torch.distributions.normal import Normal

from gaae import *

# Turn on internal checks for debugging.
# Available only from Pyro 1.3.0.
try:
   pyro.enable_validation(True)
except AttributeError:
   pass


class GeneratorModel:
   def __init__(self, generator):
      self.genr = generator

   def __call__(self, obs):
      nsmpl = obs.shape[0]
      zero = torch.zeros_like(self.genr.ref)
      one  = torch.ones_like(self.genr.ref)
      # This plate has to be present in the guide as well.
      with pyro.plate('plate_z', nsmpl):
         # Pyro sample "z" (latent variable).
         z = pyro.sample('z', dist.Normal(zero, one).to_event(1))
      # Push forward through the layers.
      mu, sd = self.genr(z)
      # Remove unobserved variables.
      mux = mu[:,:45]
      sdx = sd[:,:45]
      with pyro.plate('plate_x', nsmpl):
         # Pyro sample "x" (observed variables).
         pyro.sample('x', dist.Normal(mux, sdx).to_event(1), obs=obs)


class GeneratorGuide:
   def __init__(self, generator):
      self.genr = generator

   def __call__(self, obs):
      nsmpl = obs.shape[0]
      zero = torch.zeros_like(self.genr.ref).expand([nsmpl,-1])
      one  = torch.ones_like(self.genr.ref).expand([nsmpl,-1])
      mu = pyro.param('mu', zero)
      sd = pyro.param('sd', one, constraint=constraints.positive)
      # This plate is also present in the model.
      with pyro.plate('plate_z', nsmpl):
         # Pyro sample "z" (latent variables). We indicate that
         # the rightmost dimension is the shape of satistical events.
         pyro.sample('z', dist.Normal(mu, sd).to_event(1))


if __name__ == "__main__":

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   genr = Decoder()
   genr.load_state_dict(torch.load('gaae-decd-1000.tch'))
   genr.eval()

   data = qPCRData('data_pyro.txt', randomize=False, test=False)

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda':
      genr.cuda()

   model = GeneratorModel(genr)
   guide = GeneratorGuide(genr)

   # Declare Adam-based Stochastic Variational Inference engine.
   optim = torch.optim.Adam
   sched = pyro.optim.MultiStepLR({
      'optimizer': optim,
      'optim_args': {'lr': 0.1, 'betas': (0.90, 0.999)},
      'milestones': [1000],
      'gamma': 0.1,
   })
   svi = pyro.infer.SVI(model, guide, sched, loss=pyro.infer.Trace_ELBO())
   
   with torch.cuda.device(0):
      for batch in data.batches(btchsz=1):
         batch = batch.to(device)
         # Remove variables to predict.
         obs = batch[:,:45]
         pyro.clear_param_store()
         loss = 0
         for step in range(3000):
            loss += svi.step(obs)
            if (step+1) % 20 == 0:
               sys.stderr.write('%d: %f\n' % (step+1, loss))
               loss = 0
         # Inferred parameters.
         infmu = pyro.param('mu')
         infsd = pyro.param('sd')
         # Sample latent variables with approximate posterior.
         z = Normal(infmu, infsd).sample([1000])
         # Propagate forward and sample observable 'x'.
         with torch.no_grad():
            mu,sd = genr(z)
         for i in range(batch.shape[0]):
            x = Normal(mu[:,i,90:], sd[:,i,90:]).sample()
            orig = batch[i,:45].expand([1000, 45])
            out = torch.cat([x,orig], dim=1)
            np.savetxt(sys.stdout, out.cpu().numpy(), fmt='%.4f')
