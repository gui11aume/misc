#!/usr/bin env python

import sys

amp = dict()

with open(sys.argv[1]) as f:
   _ = next(f) # SDS version.
   _ = next(f) # Column names.
   for line in f:
      items = line.split()
      well = int(items[0])
      dRn = [float(x) for x in items[47:]]
      amp[well] = dRn
   for row in range(8):
      for col in range(12):
         code = 'ABCDEFGH'[row] + ('%02d' % (col+1))
         id384 = row * 48 + 2 * col + 25
         val = amp[9d384]
         sys.stdout.write('%s\t' % code)
         sys.stdout.write('%s\n' % '\t'.join(val))
