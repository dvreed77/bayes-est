from Pmf import *

def ProbBigger(pmf1, pmf2):
	
	total = 0.

	for v1, p1 in pmf1.Items():
		for v2, p2 in pmf2.Items():
			if v1 > v2:
				total += p1 * p2

	return total
	

six = MakePmfFromList(range(1, 7))
ten = MakePmfFromList(range(1, 11))	

for value, prob in ten.Items():
	print value, '->', prob

for value, prob in six.Items():
	print value, '->', prob	

print 'Prob ten-sider is bigger', ProbBigger(ten, six)
print 'Prob six-sider is bigger', ProbBigger(six, six)			