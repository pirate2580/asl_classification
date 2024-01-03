import numpy as np

# file to test if images loaded properly onto the npy files

loaded_data = np.load('train_y.npy')

omg = np.load('train_x.npy')
print(loaded_data[6000])
print(omg[1])