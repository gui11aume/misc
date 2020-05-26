import numpy as np
import pyro
import pyro.distributions as dist
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.constraints as constraints

from torch.distributions.normal import Normal

from aae import *

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
      #mu = torch.zeros_like(self.genr.ref).expand([nsmpl,-1])
      #sd = torch.ones_like(self.genr.ref).expand([nsmpl,-1])
      mu = torch.zeros(5, device='cuda').expand([nsmpl,-1])
      sd = torch.ones(5, device='cuda').expand([nsmpl,-1])
      # Pyro sample referenced "z" (latent variable).
      # This plate has to be present in the guide as well.
      with pyro.plate('plate_z', nsmpl):
         # Here we indicate that the rightmost dimension is
         # the shape of satistical events (4), and that we
         # want "nsmpl" independent events.
         z = pyro.sample('z', dist.Normal(mu,sd).to_event(1))
      # Push forward through the layers.
      y = self.genr(z)
      # Remove unobserved variables.
      yx = y[:,90:]
      with pyro.plate('plate_x', nsmpl):
         # Pyro sample referenced "x" (observed variables).
         pyro.sample('x', dist.Delta(yx).to_event(1), obs=obs)

class GeneratorGuide:
   def __init__(self, generator):
      self.genr = generator

   def __call__(self, obs):
      nsmpl = obs.shape[0]
      zero = torch.zeros_like(self.genr.ref).expand([nsmpl,-1])
      one  = torch.ones_like(self.genr.ref).expand([nsmpl,-1])
      mu = pyro.param('mu', zero)
      sd = pyro.param('sd', one, constraint=constraints.positive)
      # Pyro sample referenced "z" (latent variables).
      # This plate was present in the model.
      with pyro.plate('plate_z', nsmpl):
         # Here we indicate that the rightmost dimension is
         # the shape of satistical events (4), and that we
         # want "nsmpl" independent events.
         pyro.sample('z', dist.Normal(mu,sd).to_event(1))


if __name__ == "__main__":

   if sys.version_info < (3,0):
      sys.stderr.write("Requires Python 3\n")

   genr = Decoder()
   genr.load_state_dict(torch.load('aae-decd-100.tch'))

   data = qPCRData('qPCR_data.txt')

   # Do it with CUDA if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   if device == 'cuda': genr.cuda()

   genr.eval()
   model = GeneratorModel(genr)
   guide = GeneratorGuide(genr)


   # Declare Adam-based Stochastic Variational Inference engine.
   #adam_params = {'lr': 0.01, 'betas': (0.90, 0.999)}
   #opt = pyro.optim.Adam(adam_params)
   optim = torch.optim.Adam
   sched = pyro.optim.MultiStepLR({
      'optimizer': optim,
      'optim_args': {'lr': 0.05, 'betas': (0.90, 0.999)},
      'milestones': [50, 650, 750],
      'gamma': 0.1,
   })
   svi = pyro.infer.SVI(model, guide, sched, loss=pyro.infer.Trace_ELBO())
   
   for batch in data.batches(btchsz=256):
      b = batch.shape[0] # Batch size.
      d = batch.shape[1] # Dimension of the observations.
      # Pre-process the data to avoid NaN on 0s and 1.
      # Remove variable to predict.
      #idx = torch.tensor(np.delete(np.arange(X), np.arange(Y))).to(device)
      obs = batch[:,90:].to(device)
      pyro.clear_param_store()
      loss = 0
      for step in range(800):
         loss += svi.step(obs)
      # Inferred parameters.
      infmu = pyro.param('mu')
      infsd = pyro.param('sd')
      # Sample latent variables with approximate posterior.
      z = Normal(infmu, infsd).sample([128])
      # Propagate forward and sample observable 'x'.
      with torch.no_grad():
         y = genr(z)
      y = y.sum(dim=0) / 128
      np.savetxt(sys.stdout, y.cpu().numpy(), fmt='%.4f')

      break
