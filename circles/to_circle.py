import pandas as pd
import numpy as np
pi = np.pi

def map_to_circle(x,T):
	# x is 1d numpy array
	# T is periodicity
	# returns nx2 array of sin(2*pi*x/T),cos(2*pi*x/T)
	x = x.reshape(-1,1)
	x = x % T # no actual necessity, but gets x in range [0,T). Potentially helps if x >> T (much greater).
	t = x * (2*pi/T)
	X = np.concatenate((np.sin(t),np.cos(t)),axis=1)
	return X

def parse_dtime_string(dtime_string,year):
	# Input:
		# dtime_string has format:
		#  06/30  00:01:00
	# Output:
		# pandas.Timestamp

	date,time = dtime_string.strip().split()
	month, day = date.split('/')
	hour, minute, second = time.split(':')
	return pd.Timestamp(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=int(second))
def parse_dtime_strings(dtime_strings,year):
	# Input:
		# dtime_strings is 1d numpy array of dtime_strings like:
	 	# 06/30  00:01:00
	# Output:
		# None
		# modifies numpy array to pandas.Timestamp values
	dtime_strings = dtime_strings.reshape(-1,)
	for i,x in enumerate(dtime_strings):
		dtime_strings[i] = parse_dtime_string(x,year)

def timestamp_to_nums(timestamp):
	doy = timestamp.day_of_year
	dow = timestamp.day_of_week
	tod = timestamp.hour + timestamp.minute/60
	return doy,dow,tod

def timestamps_to_nums(timestamps):
	timestamps = timestamps.reshape(-1,)
	nums = np.empty((timestamps.size,3))
	for i,x in enumerate(timestamps):
		nums[i,:] = timestamp_to_nums(x)
	return nums

def timestamps_to_circles(timestamps):
	timestamps = timestamps.reshape(-1,)
	nums = timestamps_to_nums(timestamps)
	circles = np.empty((nums.shape[0],6))
	circles[:,0:2] = map_to_circle(nums[:,0],365)
	circles[:,2:4] = map_to_circle(nums[:,1],7)
	circles[:,4:6] = map_to_circle(nums[:,2],24)
	return circles

def append_position_encodings(X,wrap=False,r=2):
	L = X.shape[0]
	assert(r<=L)
	# Input:
		# X is a sequence.
		# X is L x n_x, where L is sequence length, n_x number of features.
		#  wrap is whether the first element of the sequence should be placed adjacent to the L'th, otherwise they are on opposite sides
		#  r is the reduction factor, the smaller r the more positions you get.
	# Output:
		# X with position encodings appended to the right from 2pi to the Nyquist frequency.
		# N x (n_x + n_f)
	if wrap:
		t_nr = 2*pi
	else:
		t_nr = pi
	T = L
	v = np.arange(0,L,1).reshape(L,1)
	while T>1:
		sins = np.sin(v*t_nr/T).reshape(L,1)
		coss = np.cos(v*t_nr/T).reshape(L,1)
		X = np.concatenate((X,sins,coss),axis=1)
		T /=r
	return X
