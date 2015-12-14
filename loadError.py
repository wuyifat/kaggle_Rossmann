import matplotlib.pyplot as plt

with open("error4.txt") as f:
	lines = f.readlines()[1:]

trainError = []
testError = []
col = 0

for line in lines:
	for t in line.split('\t')[1:]:
		
#		print t
		if col == 0:
			trainError.append(float(t.split(':')[1]))
		elif col == 1:
			num = t.split(':')[1]
			num = float(num.rstrip())
			testError.append(num)
		else:
			print "error"
		col += 1
		col %= 2
	#	print t[0].split(':')[1]
	#	print t[1].split(':')[1]

	"""
		for n in t.split(':'):

			print n

print trainError
print testError
print type(trainError[0])
print type(testError[0])
"""
plt.plot(trainError)
plt.plot(testError)
plt.legend(['train', 'test'])
plt.show()