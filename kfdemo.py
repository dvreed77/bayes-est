from numpy import array, newaxis, vstack, arange, zeros,\
 diag, sqrt, eye, dot, outer, c_, convolve as conv, ones, mean
from numpy.random import randn
from numpy.linalg import matrix_rank as rank, inv
import matplotlib.pyplot as plt
from scipy.signal import lfilter


# My estimate of the model
def model_est():
	dt = 1e-3

	A = array([[1, dt], [0, 1]])
	# H = array([1, 0])
	H = array([[1, 0],[0, 0]])

	B = array([(dt**2)/2, dt])
	sig_w = (1e+4)*B
	cov_w = outer(sig_w, sig_w)

	# sig_v = 1e+2
	# cov_v = sig_v**2	

	sig_v = 1e+4*array([1,1])
	cov_v = 1e+4*eye(2) #outer(sig_v, sig_v)		

	return dt, A, H, sig_w, cov_w, sig_v, cov_v

# The exact model that represents the physics of x, and the measurement of y
def model():
	dt = 1e-2

	A = array([[1, dt], [0, 1]])
	# H = array([1, 0])
	H = array([[1, 0],[0, 0]])

	B = array([(dt**2)/2, dt])
	sig_w = (1e3)*B
	cov_w = outer(sig_w, sig_w)

	# sig_v = 1e+2
	# cov_v = sig_v**2	

	sig_v = 1e2*array([1,1])
	cov_v = 1e2*eye(2) #outer(sig_v, sig_v)	

	return dt, A, H, sig_w, cov_w, sig_v, cov_v

# The exact model that represents the physics of x, and the measurement of y
def model2():
	dt = 1e-2

	A = array([[1, dt], [0, 1]])
	H = array([1, 0])[newaxis, :]
	# H = array([[1, 0],[0, 0]])

	B = array([(dt**2)/2, dt])
	sig_w = (1e+2)*B
	cov_w = outer(sig_w, sig_w)

	sig_v = 1e+2
	cov_v = sig_v**2	

	# sig_v = 1e+2*array([1,1])
	# cov_v = outer(sig_v, sig_v)	

	return dt, A, H, sig_w, cov_w, sig_v, cov_v	

# Setup the similation using the exact model
def simulation(m=model, plot=False):

	n_samples = 2000

	dt, A, H, sig_w, cov_w, sig_v, cov_v = m()

	# x_initial = c_[1, 0.1].T

	# x = x_initial
	# y = dot(H, x) + sig_v*randn()	
	t = arange(0, dt*n_samples, dt)

	x = zeros((2, len(t)+1))
	y = zeros((2, len(t)+1))

	x_initial = array([1, 0.1])
	x[:,0] = x_initial
	y[:,0] = dot(H, x_initial) + sig_v*randn()

	for i in range(1, n_samples):
		w = sig_w * randn()
		v = sig_v * randn()		

		x_tmp = dot(A, x[:, i-1]) + w
		x[:,i] = x_tmp

		y_tmp = dot(H, x[:, i]) + v
		y[:,i] = y_tmp
		
	if plot:	
		fig = plt.figure()
		ax = fig.add_subplot(111)
		
		ax.plot(t, y[0, :-1], alpha=0.2, label='Measured')
		ax.plot(t, x[0, :-1], label='Actual')

		plt.legend()
		# plt.show()

	return t, x, y	

def predict(X, P, A, Q):
	X = dot(A, X)
	P = dot(A, dot(P, A.T)) + Q
	return X,P

def update(X, Y, P, H, R):

	M = dot(H, X)
	S = dot(H, dot(P, H.T)) + R
	K = dot(P, dot(H.T, inv(S)))	
	X = X + dot(K, (Y - M))

	P = P - dot(K, dot(H, P))

	return X,P

def kalman(t, y, m):

	dt, A, H, sig_w, cov_w, sig_v, cov_v = m()

	P = diag([(1e5)**2, (1e5)**2])

	# Something weird here
	# H = H[newaxis, :]
	
	x_est = zeros((2, len(t)+1))
	x_est[:,0] = y[:, 0] #array([1,1])

	for i in range(len(t)):
		x_tmp, P = predict(x_est[:, i], P, A, cov_w)
		x_tmp, P = update(x_tmp, y[:, i], P, H, cov_v)
		x_est[:, i+1] = x_tmp

	return x_est

def kalman2(t, y, m):

	dt, A, H, sig_w, cov_w, sig_v, cov_v = m()

	Pp = diag([(1e0)**2, (1e2)**2])

	# xhatp = array([1, 0.1])[:, newaxis]

	# Estimate the initial point from the first measurement
	xhatp = array([1, 1])

	xhat = zeros((2, len(t)+1))

	H = H[newaxis, :]

	for k in range(len(t)):
		# w[:, k] = (B * sqrt(vara) * randn()).squeeze()
		# v[:, k] = sqrt(R) * randn()

		# x[:, k+1] = (A.dot(x[:, k][:, newaxis]) + w[:, k][:, newaxis]).squeeze()
		# y[:, k] = (C.dot(x[:, k][:, newaxis]) + v[:, k][:, newaxis]).squeeze()

		Y = y[:, k]

		# Update
		M = dot(H, xhatp)
		S = dot(H, dot(Pp, H.T)) + cov_v
		K = dot(Pp, H.T) * 1./S
		# K = Pp.dot(H.T) * 1./(H.dot(Pp).dot(H.T) + cov_v)
		xhat[:, k] = dot(A, xhatp) + dot(K, (Y - M))

		import pdb; pdb.set_trace()


		P = Pp - dot(K, dot(H, Pp))

		# Predict
		xhatp = A.dot(xhat[:, k])
		Pp = A.dot(P).dot(A.T) + cov_w


		# K = Pp.dot(H.T) * 1./(H.dot(Pp).dot(H.T) + cov_v)
		# xhat[:, k] = (A.dot(xhatp) + K.dot(y[:, k][:, newaxis] - H.dot(xhatp))).squeeze()
		# P = (eye(2) - K.dot(H)).dot(Pp)
		# xhatp = A.dot(xhat[:, k][:, newaxis])
		# Pp = A.dot(P).dot(A.T) + cov_w

	return xhat	


def my_mean(y, step):

	x = zeros(len(y))

	for i in range(step, len(y)):
		i1 = i - step
		i2 = i
		x[i] = mean(y[i1:i2])

	return x	


def main2():

	my_model = model


	t, x, y = simulation(m=my_model)
	x_est = kalman(t, y, my_model)

	x_mean = conv(ones(100)/100, y[0, :-1], 'same')

	m2 = my_mean(y[0, :-1], 100)

	m3 = lfilter(ones(100)/100, 1, y[0, :-1])
	# import pdb; pdb.set_trace()

	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.plot(t, y[0, :-1], alpha=0.2, label='Measured')
	ax1.plot(t, x[0, :-1], label='Actual')
	ax1.plot(t, x_est[0, :-1], label='Estimated')
	# ax1.plot(t, x_mean)
	ax1.plot(t, m2, label='Moving Average')
	ax1.plot(t, m3, label='Moving Average')

	plt.legend()	

	# # fig = plt.figure()
	# ax2 = fig.add_subplot(212)

	# ax2.plot(t, y[1, :-1], alpha=0.2, label='Measured')
	# ax2.plot(t, x[1, :-1], label='Actual')
	# ax2.plot(t, x_est[1, :-1], label='Estimated')

	# plt.legend()

	k_mean = mean((x[0,:] - x_est[0,:])**2)

	e1 = mean((x[0,:-1] - y[0,:-1])**2)/k_mean
	e2 = mean((x[0,:-1] - m2)**2)/k_mean
	e3 = mean((x[0,:-1] - m3)**2)/k_mean

	b1 = 'Better' if e1 < 1 else 'Worse'
	b2 = 'Better' if e2 < 1 else 'Worse'
	b3 = 'Better' if e3 < 1 else 'Worse'

	print "Measured Error:", e1, b1
	print "My Moving Avg:", e2, b2
	print "FIR Moving Avg:", e3, b3











def kfdemo():

	T = 1e-2
	A = array([[1, T], [0, 1]])
	B = array([(T**2)/2, T])[:, newaxis]
	C = array([1, 0])[newaxis, :]
	# C = array([[1, 0], [0, 0]])

	vara = (1e+2)**2

	Q = vara*B.dot(B.T)
	R = (1e+2)**2

	n = A.shape[0]
	m = C.shape[0]

	O = vstack([C, C.dot(A)])

	print O
	print 'Size(O) = %ix%i' % (O.shape[0], O.shape[1])
	print 'Rank(O) = %i' % (rank(O))

	max_iter = 2e3

	t = arange(0,T*max_iter, T)

	w = zeros((n, max_iter))
	v = zeros((m, max_iter))

	x = zeros((n, max_iter+1))
	y = zeros((m, max_iter))

	xhat = zeros((n, max_iter))
	x[:, 0] = [1, 0.1]

	Pp = diag([(1e0)**2, (1e2)**2])

	xhatp = sqrt(Pp).dot(randn(2,1)) + x[:, 0][:, newaxis];

	# import pdb; pdb.set_trace()

	for k in range(int(max_iter)):
		w[:, k] = (B * sqrt(vara) * randn()).squeeze()
		v[:, k] = sqrt(R) * randn()

		x[:, k+1] = (A.dot(x[:, k][:, newaxis]) + w[:, k][:, newaxis]).squeeze()
		y[:, k] = (C.dot(x[:, k][:, newaxis]) + v[:, k][:, newaxis]).squeeze()

		K = Pp.dot(C.T) * 1./(C.dot(Pp).dot(C.T) + R)
		xhat[:, k] = (A.dot(xhatp) + K.dot(y[:, k][:, newaxis] - C.dot(xhatp))).squeeze()
		import pdb; pdb.set_trace()

		P = (eye(n) - K.dot(C)).dot(Pp)
		xhatp = A.dot(xhat[:, k][:, newaxis])
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

if __name__ == '__main__':
	kfdemo()	