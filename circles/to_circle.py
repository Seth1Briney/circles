import pandas as pd
import numpy as np
from numpy import pi, log

def map_to_circle(x,T):
	# x is 1d numpy array
	# T is periodicity
	# returns nx2 array of sin(2*pi*x/T),cos(2*pi*x/T)
	x = np.array(x).reshape(-1,1)
	x = x % T # no actual necessity, but gets x in range [0,T). Potentially helps if x >> T (much greater).
	t = x * (2*pi/T)
	X = np.concatenate((np.sin(t),np.cos(t)),axis=1)
	return X

# show = False
def parse_dtime_string(dtime_string,year):
	# global show
	# Input:
		# dtime_string has format:
		#  06/30  00:01:00
	# Output:
		# pandas.Timestamp

	date,time = dtime_string.strip().split()
	month, day = date.split('/')
	hour, minute, second = time.split(':')
	hour = int(hour)
	# if hour==23:
	# 	show=True
	if hour==24:
		hour = 23
		minute = int(minute)
		assert(minute==0)
		minute=59
		dts = pd.Timestamp(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=int(second))
		dts = dts + pd.Timedelta(minutes=1)
	else:
		dts = pd.Timestamp(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=int(second))

	# if show:
	# 	print(dts)
	# 	input()
	return dts
def parse_dtime_strings(dtime_strings,year):
	# Input:
		# dtime_strings is 1d numpy array of dtime_strings like:
	 	# 06/30  00:01:00
	# Output:
		# None
		# modifies numpy array to pandas.Timestamp values
	dtime_strings = np.array(dtime_strings).reshape(-1,)
	for i,x in enumerate(dtime_strings):
		dtime_strings[i] = parse_dtime_string(x,year)
	return dtime_strings

def timestamp_to_nums(timestamp):
	doy = timestamp.day_of_year
	dow = timestamp.day_of_week
	tod = timestamp.hour + timestamp.minute/60
	return doy,dow,tod

def timestamps_to_nums(timestamps):
	timestamps = np.array(timestamps).reshape(-1,)
	nums = np.empty((timestamps.size,3))
	for i,x in enumerate(timestamps):
		nums[i,:] = timestamp_to_nums(x)
	return nums

def timestamps_to_circles(timestamps):
	timestamps = np.array(timestamps).reshape(-1,)
	nums = timestamps_to_nums(timestamps)
	circles = np.empty((nums.shape[0],6))
	circles[:,0:2] = map_to_circle(nums[:,0],365)
	circles[:,2:4] = map_to_circle(nums[:,1],7)
	circles[:,4:6] = map_to_circle(nums[:,2],24)
	return circles

def strings_to_circles(strings,year):
	return timestamps_to_circles(parse_dtime_strings(strings,year))

def append_position_encodings(X,wrap=False,r=2):
	L = X.shape[0]
	assert(1<r<=L)
	# Input:
		# X is a sequence.
		# X is L x n_x, where L is sequence length, n_x number of features.
		#  wrap is whether the first element of the sequence should be placed adjacent to the L'th, otherwise they are on opposite sides
		#  r is the reduction factor, the smaller r the more positions you get.
		# r > 1
	# Output:
		# X with position encodings appended to the right from 2pi to the Nyquist frequency.
		# N x (n_x + n_f)
	if wrap:
		t_nr = 2*pi
	else:
		t_nr = pi
	T = L
	v = np.arange(0,L,1).reshape(L,1)
	while T>2:
		sins = np.sin(v*t_nr/T).reshape(L,1)
		coss = np.cos(v*t_nr/T).reshape(L,1)
		X = np.concatenate((X,sins,coss),axis=1)
		T /=r
	return X

def append_position_encodings_vectorized(X,wrap=False,r=2):
	L = X.shape[0]
	assert(r<=L)
	if wrap:
		t_nr = 2*pi
	else:
		t_nr = pi
	T = L
	v = np.arange(0,L,1).reshape(L,1).reshape(-1,1)
	# Solve for N:
		# T / r^N = 2
		# T = 2r^N
		# T/2 = r^N
		# log(T/2) = N log(r)
		# N = log(T/2)/log(r)
	N = int(log(T/2)/log(r))
	n = np.arange(0,N+1,1)
	t = (v * (r**n)) * (t_nr / T)

	X = np.concatenate(( X , \
		np.sin(t), \
		np.cos(t)  \
		),axis=1)
	return X

def append_all_position_encodings(X,wrap=False):
	L = X.shape[0]

	v = np.arange(0,L,1).reshape(L,1)

	if wrap:
		# L-1 -> 2pi
		# 2pi (L-1) / T = 2pi
		# T = (L-1)
		T = np.arange(2,L+1,1)
	else:
		# L-1 -> pi
		# 2pi (L-1) / T = pi
		# T = 2(L-1)
		T = np.arange(2,2*(L-1)+1,1)

	T = T[::-1]

	t = (v / T) * (2*pi)

	X = np.concatenate(( X , \
		np.sin(t), \
		np.cos(t)  \
		),axis=1)
	return X

def print_array(x):
	# 2d array, x
	length = 5
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			num = np.round(x[i,j],2)
			if num>=0:
				num=' '+str(num)
			else:
				num = str(num)
			num = num+' '*(length-len(num))
			print(num,end=' ')
		print()