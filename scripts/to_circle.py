'''
Example of how to use to_circle module.
'''

# def timestamps_to_circles(ts):
	# ts is array of pd.Timestamp

# if __name__=='__main__':
	# Timestamp(ts_input=<object object at 0x103690f60>, freq=None, tz=None, unit=None, year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None, nanosecond=None, tzinfo=None, *, fold=None)
d = np.load('2017_1.npz',allow_pickle=True) # allow pickle because dates saved as strings
# d.files
# ['D', 'To', 'Ts', 'EL', 'source', 'year']
dates = d.get('D')
ts = pd.Timestamp
for i in range(3):
	print(dates[i])
	print(parse_dtime_string(dates[i],2017))

parse_dtime_strings(dates,2017)

circles = timestamps_to_circles(dates)
print(circles)
circles_pose = append_position_encodings(circles)