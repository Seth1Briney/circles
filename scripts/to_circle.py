'''
Example of how to use to_circle module.
'''
import circles
import numpy as np

# def timestamps_to_circles(ts):
	# ts is array of pd.Timestamp

# if __name__=='__main__':
	# Timestamp(ts_input=<object object at 0x103690f60>, freq=None, tz=None, unit=None, year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None, nanosecond=None, tzinfo=None, *, fold=None)
d = np.load('2017_1.npz',allow_pickle=True) # allow pickle because dates saved as strings
# d.files
# ['D', 'To', 'Ts', 'EL', 'source', 'year']
dates = d.get('D')
# ts = pd.Timestamp
# for i in range(3):
# 	print(dates[i])
# 	print(parse_dtime_string(dates[i],2017))

dates = dates[:6]

circles.to_circle.parse_dtime_strings(dates,2017)

cs = circles.to_circle.timestamps_to_circles(dates)
# print(circles)
r = 2
circles_pose = circles.to_circle.append_position_encodings(cs,r=r)

print(circles_pose[:,6:])

circles_pv = circles.to_circle.append_position_encodings_vectorized(cs,r=r)
err = circles_pose-circles_pv
print(np.std(err))
print(np.where(err>.01))
print(err[np.where(err>.01)])

circles.to_circle.print_array(circles_pv)
print()
circles.to_circle.print_array(circles_pose)
print('note the results are the same, but the ordering of the columns is different. That is fine.')

all_pose = circles.to_circle.append_all_position_encodings(cs[:3],)
circles.to_circle.print_array(all_pose[:,6:])
all_pose = circles.to_circle.append_all_position_encodings(cs[:3],wrap=True)
circles.to_circle.print_array(all_pose[:,6:])
# print(np.linalg.norm(circles_pose-circles_pv))
# print(np.linalg.norm(circles_pose-circles_pv))