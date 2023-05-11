import numpy as np

def loadData():
  x = np.random.rand(16000,1)*2*np.pi-np.pi 
  y = np.sin(x)
  return x , y


def loadOutlierData():
 
  x = np.random.rand(16000,1)*2*np.pi-np.pi 
  y = np.sin(x)
  
  # Addings outliers
  y[np.random.randint(0,16000,3)]=1000.

  return x , y