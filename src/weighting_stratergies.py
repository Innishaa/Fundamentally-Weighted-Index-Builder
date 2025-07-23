import numpy as np
def equal_weight(n):
   return np.ones(n) / n
def market_cap_weight(mcaps):
   return mcaps / np.sum(mcaps)
def ff_market_cap_weight(mcaps, ff_factors):
   ff_mcap = mcaps * ff_factors
   return ff_mcap / np.sum(ff_mcap)