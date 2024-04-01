
import numpy as np
import math
import sys
import timeit

# Important Constants
EXPERIMENT_COUNT = 10
RANGE_OF_INT_VALUES = 1000
DIM_BLOCK = 32
RANGE_LOW = 100
RANGE_HIGH = 2001
RANGE_STEP = 100


device_time_dict = {}
device_time_std_dict = {}
print("size, host mean")   #print header
for size in range(RANGE_LOW, RANGE_HIGH, RANGE_STEP):
  inputA = np.random.randint(RANGE_OF_INT_VALUES, size=size)
  inputB = np.random.randint(RANGE_OF_INT_VALUES, size=size)
  output = np.array([0])


host_time_dict = {}
host_time_std_dict = {}
for size in range(RANGE_LOW, RANGE_HIGH, RANGE_STEP):
  inputA = np.random.randint(RANGE_OF_INT_VALUES, size=size)
  inputB = np.random.randint(RANGE_OF_INT_VALUES, size=size)
  """ Host experiment """
  host_time_tot = 0
  host_std_sum = 0
  for z in range(EXPERIMENT_COUNT):
    # start time
    time_started = timeit.default_timer()
    for element in range(len(inputA)):
      output += inputA[element]*inputB[element]
    # end time
    time_ended = timeit.default_timer()
    time_distance = time_ended - time_started
    host_time_tot += time_distance
    host_std_sum += time_distance**2

  host_time_ave = host_time_tot/EXPERIMENT_COUNT
  host_time_dict[size] = host_time_ave
  host_std = math.sqrt(host_std_sum/EXPERIMENT_COUNT - host_time_ave**2)
  host_time_std_dict[size] = host_std
  # print(output)


for size in range(RANGE_LOW, RANGE_HIGH, RANGE_STEP):
  print("%d, %.8f"%(size, host_time_dict[size]))     # display result time