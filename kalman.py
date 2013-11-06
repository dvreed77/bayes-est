import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def generate(A, B, R, vara, C, n):
	w = []
	v = []
	x = [[1, 0.1]]
	y = []
	for k in np.arange(n):
		w.append(B * np.sqrt(vara) * np.random.randn())
		v.append(np.sqrt(R) * np.random.randn())

		y.append(C.dot(x[-1]) + v[-1])
		x.append(A.dot(x[-1]) + w[-1])

	return np.array(x), np.array(y)

class Kalman(object):
	"""docstring for Kalman"""
	def __init__(self, A, Q, C, Pp, R):
		super(Kalman, self).__init__()
		self.A = A
		self.Q = Q
		self.C = C
		self.Pp = Pp
		self.R = R
		self.state = []
		self.xhatp = None

	def update(self, y):

		C = self.C
		Pp = self.Pp
		R = self.R
		A = self.A
		if self.xhatp is None:
			self.xhatp = y

		xhatp = self.xhatp
		# M = dot(H, X)
		# S = dot(H, dot(P, H.T)) + R
		# K = dot(P, dot(H.T, inv(S)))	
		# X = X + dot(K, (Y - M))

		# P = P - dot(K, dot(H, P))

		# return X,P

		S = C.dot(Pp).dot(C.T) + R

		# K = Pp.dot(C.T) * 1./(C.dot(Pp).dot(C.T) + R)
		self.K = Pp.dot(C.T).dot(la.inv(S))
		self.state.append(A.dot(xhatp) + self.K.dot(y - C.dot(xhatp)))

	def predict(self):
		K = self.K
		n = 2
		C = self.C
		Pp = self.Pp
		Q = self.Q
		A = self.A

		# X = dot(A, X)
		# P = dot(A, dot(P, A.T)) + Q
		# return X,P

		# xhatp = A.dot(xhat[:, k])
		xhatp = A.dot(self.state[-1])

		P = (np.eye(n) - K.dot(C)).dot(Pp)		
		Pp = A.dot(P).dot(A.T) + Q

		# STOP

def run():
	# timestep
	T = 1e-2
	# model
	A = np.array([[1, T], [0, 1]])

	# acceleration term, part of process noise
	# the noise we are adding is a random acceleration
	vara = (1e+2)**2
	B = np.array([(T**2)/2, T])

	# observation matrix
	C = np.array([[1, 0],[0, 0]])

	

	# Q = vara*B.dot(B.T)
	Q = vara*np.outer(B,B)

	# Measurement Noise
	R = (1e+2)**2

	Pp = np.diag([(1e0)**2, (1e2)**2])

	# STOP

	x,y = generate(A, B, R, vara, C, 2e3)

	k = Kalman(A, Q, C, Pp, R)

	for ii in np.arange(2e3):
		k.update(y[ii])
		k.predict()

	return x,y,k


def kfdemo():

	T = 1e-2
	A = np.array([[1, T], [0, 1]])
	B = np.array([(T**2)/2, T])
	C = np.array([[1, 0],[0, 0]])
	# C = array([[1, 0], [0, 0]])

	vara = (1e+2)**2

	Q = vara*B.dot(B.T)
	R = (1e+2)**2

	n = A.shape[0]
	m = 2

	max_iter = 2e3

	t = np.arange(0,T*max_iter, T)

	w = np.zeros((n, max_iter))
	v = np.zeros((m, max_iter))

	x = np.zeros((n, max_iter+1))
	y = np.zeros((m, max_iter))

	xhat = np.zeros((n, max_iter))
	x[:, 0] = [1, 0.1]

	Pp = np.diag([(1e0)**2, (1e2)**2])

	xhatp = x[:, 0]

	# import pdb; pdb.set_trace()

	for k in range(int(max_iter)):
		w[:, k] = B * np.sqrt(vara) * np.random.randn()
		v[:, k] = np.sqrt(R) * np.random.randn()

		x[:, k+1] = A.dot(x[:, k]) + w[:, k]
		y[:, k] = C.dot(x[:, k]) + v[:, k]

		K = Pp.dot(C.T) * 1./(C.dot(Pp).dot(C.T) + R)
		xhat[:, k] = A.dot(xhatp) + K.dot(y[:, k] - C.dot(xhatp))
		# import pdb; pdb.set_trace()

		P = (np.eye(n) - K.dot(C)).dot(Pp)
		xhatp = A.dot(xhat[:, k])
		Pp = A.dot(P).dot(A.T) + Q

	fig = plt.figure()

	ax1 = fig.add_subplot(221)
	ax1.plot(t,x[0,0:-1], 'r')
	ax1.plot(t,xhat[0,:],'b')
	# ax1.plot(t,y[0,:],'g')
	plt.xlabel('Time')
	plt.ylabel('x_1(t) (m)')
	plt.title('Red: Actual, Blue: Estimated')
	# plt.ylim([0, 3])

	ax2 = fig.add_subplot(222)
	ax2.plot(t,x[1,0:-1], 'r')
	ax2.plot(t,xhat[1,:],'b')
	plt.xlabel('Time')
	plt.ylabel('x_2(t) (m/s)')
	plt.title('Red: Actual, Blue: Estimated')

	print xhat.shape, t.shape

	ax3 = fig.add_subplot(223)
	ax3.plot(t,x[0,0:-1] - xhat[0,:], 'g')
	plt.xlabel('Time')
	plt.ylabel('$e_1(t)$ (m)')

	ax4 = fig.add_subplot(224)
	ax4.plot(t,x[1,0:-1] - xhat[1,:], 'g')
	plt.xlabel('Time')
	plt.ylabel('e_2(t) (m)')

	plt.show()