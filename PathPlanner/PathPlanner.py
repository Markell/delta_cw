# =================================================================================================
# -- imports --------------------------------------------------------------------------------------
# =================================================================================================

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, tan, pi


# =================================================================================================
# -- simple math helpers --------------------------------------------------------------------------
# =================================================================================================

def tand(theta):
	return tan(theta * pi / 180)


def sind(theta):
	return sin(theta * pi / 180)


def cosd(theta):
	return cos(theta * pi / 180)


# =================================================================================================
# -- point to point path planning class -----------------------------------------------------------
# =================================================================================================

class PathPlannerPTP:
	def __init__(self, theta_initial, theta_final, sampling_frequency=100):
		self.theta_i 			= theta_initial
		self.theta_f 			= theta_final
		self.sampling_frequency = sampling_frequency

	def ptp_polynomial5th(self):

		# time array (n+1 time instances where n is the sampling frequency)
		t = np.array(range(0, self.sampling_frequency + 1)) / self.sampling_frequency

		# calculating s and theta (interpolating polynomial)
		s 			= 6*t**5 - 15*t**4 + 10*t**3
		theta 		= np.array(self.theta_i) + np.array(self.theta_f - self.theta_i)*s

		# calculating s and theta | 1st Differential
		s_dot 		= 30*t**4 - 60*t**3 + 30*t**2
		theta_dot 	= np.array(self.theta_f - self.theta_i)*s_dot

		# calculating s and theta | 2nd Differential
		s_ddot 		= 120*t**3 - 180*t**2 + 60*t
		theta_ddot 	= np.array(self.theta_f - self.theta_i)*s_ddot

		# calculating s and theta | 3rd Differential
		s_dddot 	= -360*t**2 - 360*t + 60
		theta_dddot = np.array(self.theta_f - self.theta_i)*s_dddot

		return (t, theta, theta_dot, theta_ddot, theta_dddot)

	def ptp_polynomial7th(self):

		# time array (n+1 time instances where n is the sampling frequency)
		t = np.array(range(0, self.sampling_frequency + 1)) / self.sampling_frequency

		# calculating s and theta (interpolating polynomial)
		s 			= -20*t**7 + 70*t**6 - 84*t**5 + 35*t**4
		theta 		= self.theta_i + (self.theta_f - self.theta_i)*s

		# calculating s and theta | 1st Differential
		s_dot 		= -140*t**6 + 420*t**5 - 420*t**4 + 140*t**3
		theta_dot 	= (self.theta_f - self.theta_i)*s_dot

		# calculating s and theta | 2nd Differential
		s_ddot 		= -840*t**5 + 2100*t**4 - 1680*t**3 + 420*t**2
		theta_ddot 	= (self.theta_f - self.theta_i)*s_ddot

		# calculating s and theta | 3rd Differential
		s_dddot 	= -4200*t**4 + 8400*t**3 - 5040*t**2 + 840*t**1
		theta_dddot = (self.theta_f - self.theta_i)*s_dddot

		return (t, theta, theta_dot, theta_ddot, theta_dddot)

	def ptp_polynomial9th(self):

		# time array (n+1 time instances where n is the sampling frequency)
		t = np.array(range(0, self.sampling_frequency + 1)) / self.sampling_frequency

		# calculating s and theta (interpolating polynomial)
		s 			= 70*t**9 - 315*t**8 + 540*t**7 - 420*t**6 + 126*t**5
		theta 		= self.theta_i + (self.theta_f - self.theta_i)*s

		# calculating s and theta | 1st Differential
		s_dot 		= 630*t**8 - 2520*t**7 + 3780*t**6 - 2520*t**5 + 630*t**4
		theta_dot 	= (self.theta_f - self.theta_i)*s_dot

		# calculating s and theta | 2nd Differential
		s_ddot 		= 5040*t**7 - 17640*t**6 + 22680*t**5 - 12600*t**4 + 2520*t**3
		theta_ddot 	= (self.theta_f - self.theta_i)*s_ddot

		# calculating s and theta | 3rd Differential
		s_dddot 	= 35280*t**6 - 105840*t**5 + 113400*t**4 - 50400*t**3 + 7560*t**2
		theta_dddot = (self.theta_f - self.theta_i)*s_dddot

		return (t, theta, theta_dot, theta_ddot, theta_dddot)

	def ptp_bangbang(self):
		# time array (n+1 time instances where n is the sampling frequency)
		t 			= np.array(range(0, self.sampling_frequency + 1)) / self.sampling_frequency
		theta 		= np.zeros_like(t)
		theta_dot 	= np.zeros_like(t)
		theta_ddot 	= np.zeros_like(t)

		for i, time in enumerate(t):
			if time <= 0.5:  		# Phase 1: Constant Acceleration
				theta[i] 		= self.theta_i + 2 * (self.theta_f - self.theta_i) * time**2
				theta_dot[i] 	= 4 * (self.theta_f - self.theta_i) * time
				theta_ddot[i] 	= 4 * (self.theta_f - self.theta_i)

			else:  					# Phase 2: Constant Deceleration
				theta[i] 		= 	0.5 * (self.theta_i + self.theta_f) + 2 * (self.theta_f - self.theta_i) \
									* (time - 0.5) + 2 * (self.theta_i - self.theta_f) * (time - 0.5)**2
				theta_dot[i] 	= 2 * (self.theta_f - self.theta_i) * (1 - 2 * (time - 0.5))
				theta_ddot[i] 	= -4 * (self.theta_f - self.theta_i)

		return (t, theta, theta_dot, theta_ddot, None)

	def ptp_trapezoidal(self):

		# time array (n+1 time instances where n is the sampling frequency)
		t = np.array(range(0, self.sampling_frequency + 1)) / self.sampling_frequency

		total_time 	= 1  												# total time normalized to 1
		T 			= total_time / 3  									# each phase time duration
		v_max 		= (self.theta_f - self.theta_i) / (total_time - T)  # maximum velocity
		a 			= 3 * v_max  										# acceleration

		theta 		= np.zeros_like(t)
		theta_dot 	= np.zeros_like(t)
		theta_ddot 	= np.zeros_like(t)

		for i, time in enumerate(t):
			if 		time <= T:  						# Phase 1: Acceleration
				theta_dot[i] 	= a * time
				theta_ddot[i] 	= a
				theta[i] 		= self.theta_i + 0.5 * a * time**2
			elif 	time <= 2 * T:  					# Phase 2: Constant Velocity
				theta_dot[i] 	= v_max
				theta_ddot[i] 	= 0
				theta[i] 		= self.theta_i + 0.5 * a * T**2 + v_max * (time - T)
			else:  										# Phase 3: Deceleration
				theta_dot[i] 	= -a * (time - 2 * T) + v_max
				theta_ddot[i] 	= -a
				theta[i] 		= (self.theta_i + 0.5 * a * T**2 + v_max * T +
									v_max * (time - 2 * T) - 0.5 * a * (time - 2 * T)**2)

		return (t, theta, theta_dot, theta_ddot, None)

	def ptp_scurve(self):
		# time array (n+1 time instances where n is the sampling frequency)
		t 			= np.array(range(0, self.sampling_frequency + 1)) / self.sampling_frequency
		total_time 	= 1  	# total time normalized to 1
		T 			= total_time / 7  # each segment time duration

		v_max = (self.theta_f - self.theta_i) / (4 * T)
		a_max = v_max / (T * 2)
		j_max = a_max / T

		theta 		= np.zeros_like(t)
		theta_dot 	= np.zeros_like(t)
		theta_ddot 	= np.zeros_like(t)
		theta_dddot = np.zeros_like(t)

		for i, time in enumerate(t):
			if time <= T:                       # Phase 1: Increasing Acceleration
				theta_dddot[i]  = j_max
				theta_ddot[i]   = j_max*time
				theta_dot[i]    = 0.5*j_max*time**2
				theta[i]        = (self.theta_i) + 1/6*j_max*time**3
			elif time <= 2 * T:                 # Phase 2: Constant Acceleration
				theta_dddot[i]  = 0
				theta_ddot[i]   = a_max
				theta_dot[i]    = (0.5*a_max*T) + a_max*(time - T)
				theta[i]        = (self.theta_i + (1/12)*v_max*T) + (0.5*a_max*T)*(time - T) + 0.5*a_max*(time - T)**2
			elif time <= 3 * T:                 # Phase 3: Decreasing Acceleration
				theta_dddot[i]  = - j_max
				theta_ddot[i]   = a_max - j_max*(time - 2*T)
				theta_dot[i]    = (1.5*a_max*T) + (a_max*(time - 2*T) - 0.5*j_max*(time - 2*T)**2)
				theta[i]        = (self.theta_i + (7/12)*v_max*T) + (1.5*a_max*T)*(time - 2*T) + \
									(0.5*a_max*(time - 2*T)**2 - (1/6)*j_max*(time - 2*T)**3)
			elif time <= 4 * T:                 # Phase 4: Constant Velocity
				theta_dddot[i]  = 0
				theta_ddot[i]   = 0
				theta_dot[i]    = v_max
				theta[i]        = (self.theta_i + 1.5*v_max*T) + v_max*(time - 3*T)
			elif time <= 5 * T:                 # Phase 5: Increasing Deceleration
				theta_dddot[i]  = - j_max
				theta_ddot[i]   = - j_max*(time - 4*T)
				theta_dot[i]    = v_max - 0.5*j_max*(time - 4*T)**2
				theta[i]        = (self.theta_i + 2.5*v_max*T) + v_max*(time - 4*T) - (1/6)*j_max*(time - 4*T)**3
			elif time <= 6 * T:                 # Phase 6: Constant Deceleration
				theta_dddot[i]  = 0
				theta_ddot[i]   = - a_max
				theta_dot[i]    = (3/4)*v_max - a_max*(time - 5*T)
				theta[i]        = (self.theta_i + (41/12)*v_max*T) + (3/4)*v_max*(time - 5*T) - 0.5*a_max*(time - 5*T)**2
			else:                               # Phase 7: Decreasing Deceleration
				theta_dddot[i]  = j_max
				theta_ddot[i]   = - a_max + j_max*(time - 6*T)
				theta_dot[i]    = ((1/4)*v_max) + (- a_max*(time - 6*T) + 0.5*j_max*(time - 6*T)**2)
				theta[i]        = (self.theta_i + (47/12)*v_max*T) + ((1/4)*v_max)*(time - 6*T) + \
									(- 0.5*a_max*(time - 6*T)**2 + (1/6)*j_max*(time - 6*T)**3)
				# this is equal to 4.V_max.T

		return (t, theta, theta_dot, theta_ddot, theta_dddot)

	def plot(self, results, method_name, _format='.pdf', _file_path='./results - point to point/'):
		_plot_trajectory(results, method_name, _format, _file_path, labelsize=22)


# =================================================================================================
# -- multi point path planning class --------------------------------------------------------------
# =================================================================================================

class PathPlannerMLTP:
	def __init__(self, path, sampling_frequency=100):
		'''
		A Trajectory Generation Class based on a given path
		'''

		# primary values and vectors
		self.path 				= np.array(path)
		self.n 					= len(self.path) - 1
		# number of overall trajectory samples
		self.sampling_frequency = sampling_frequency
		self.sampling_number 	= sampling_frequency * self.n + 1
		# time vector corresponding to path points
		self.t_path				= np.linspace(0, 1, self.n + 1)
		# vector of time intervals for the t_path
		self.T_path 			= self.t_path[1:self.n+1] - self.t_path[0:self.n]

	def mltp_ptpmethods(self, method_name):

		t      		= np.linspace(0, 1, self.sampling_number)
		theta      	= np.zeros_like(t)
		theta_dot  	= np.zeros_like(t)
		theta_ddot  = np.zeros_like(t)
		theta_dddot = np.zeros_like(t)

		for i in range(len(self.path)-1):
			path_planner = PathPlannerPTP(self.path[i], self.path[i+1], sampling_frequency=self.sampling_frequency)

			# init
			(_t, _theta, _theta_dot, _theta_ddot, _theta_dddot) = (None, None, None, None, None)

			if method_name == "ptp_polynomial5th":
				(_t, _theta, _theta_dot, _theta_ddot, _theta_dddot) = path_planner.ptp_polynomial5th()
			elif method_name == "ptp_polynomial7th":
				(_t, _theta, _theta_dot, _theta_ddot, _theta_dddot) = path_planner.ptp_polynomial7th()
			elif method_name == "ptp_polynomial9th":
				(_t, _theta, _theta_dot, _theta_ddot, _theta_dddot) = path_planner.ptp_polynomial9th()
			elif method_name == "ptp_bangbang":
				(_t, _theta, _theta_dot, _theta_ddot, _theta_dddot) = path_planner.ptp_bangbang()
			elif method_name == "ptp_trapezoidal":
				(_t, _theta, _theta_dot, _theta_ddot, _theta_dddot) = path_planner.ptp_trapezoidal()
			elif method_name == "ptp_scurve":
				(_t, _theta, _theta_dot, _theta_ddot, _theta_dddot) = path_planner.ptp_scurve()

			theta[i*self.sampling_frequency:(i+1)*self.sampling_frequency+1] 		= _theta
			theta_dot[i*self.sampling_frequency:(i+1)*self.sampling_frequency+1] 	= _theta_dot
			theta_ddot[i*self.sampling_frequency:(i+1)*self.sampling_frequency+1] 	= _theta_ddot
			theta_dddot[i*self.sampling_frequency:(i+1)*self.sampling_frequency+1] 	= _theta_dddot

		if method_name in ["ptp_trapezoidal", "ptp_bangbang"]:
			theta_dddot = None

		return t, theta, theta_dot, theta_ddot, theta_dddot

	def mltp_polynomial7th_4point(self):

		if len(self.path) != 4:
			print("The length of the path is not equal to four")
			return None

		theta0 = self.path[0]
		theta1 = self.path[1]
		theta2 = self.path[2]
		theta3 = self.path[3]
		t      = np.linspace(0, 1, self.sampling_number)

		a0 = theta0
		a1 = 0
		a2 = 0
		a3 = 182.25 * theta1 - 134.875 * theta0 - 91.125 * theta2 + 43.75 * theta3
		a4 = 548.25 * theta0 - 820.125 * theta1 + 546.75 * theta2 - 274.875 * theta3
		a5 = 1366.875 * theta1 - 856.5 * theta0 - 1093.5 * theta2 + 583.125 * theta3
		a6 = 600.75 * theta0 - 1002.375 * theta1 + 911.25 * theta2 - 509.625 * theta3
		a7 = 273.375 * theta1 - 158.625 * theta0 - 273.375 * theta2 + 158.625 * theta3

		theta 		= a7 * t**7 + a6 * t**6 + a5 * t**5 + a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0
		theta_dot 	= 7 * a7 * t**6 + 6 * a6 * t**5 + 5 * a5 * t**4 + 4 * a4 * t**3 + 3 * a3 * t**2 + 2 * a2 * t + a1
		theta_ddot 	= 42 * a7 * t**5 + 30 * a6 * t**4 + 20 * a5 * t**3 + 12 * a4 * t**2 + 6 * a3 * t + 2 * a2
		theta_dddot = 210 * a7 * t**4 + 120 * a6 * t**3 + 60 * a5 * t**2 + 24 * a4 * t + 6 * a3

		return t, theta, theta_dot, theta_ddot, theta_dddot

	def mltp_polynomial9th_4point(self):

		if len(self.path) != 4:
			print("The length of the path is not equal to four")
			return None

		theta0 = self.path[0]
		theta1 = self.path[1]
		theta2 = self.path[2]
		theta3 = self.path[3]
		t      = np.linspace(0, 1, self.sampling_number)

		a0 = theta0
		a1 = 0
		a2 = 0
		a3 = 0
		a4 = 820.125 * theta1 - 641.9375 * theta0 - 410.0625 * theta2 + 231.875 * theta3
		a5 = 3315.5625 * theta0 - 4510.6875 * theta1 + 2870.4375 * theta2 - 1675.3125 * theta3
		a6 = 9841.5 * theta1 - 6926.875 * theta0 - 7381.125 * theta2 + 4466.5 * theta3
		a7 = 7270.625 * theta0 - 10661.625 * theta1 + 9021.375 * theta2 - 5630.375 * theta3
		a8 = 5740.875 * theta1 - 3822.1875 * theta0 - 5330.8125 * theta2 + 3412.125 * theta3
		a9 = 803.8125 * theta0 - 1230.1875 * theta1 + 1230.1875 * theta2 - 803.8125 * theta3

		theta      	= a9 * t**9 + a8 * t**8 + a7 * t**7 + a6 * t**6 + a5 * t**5 + a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0
		theta_dot  	= 9 * a9 * t**8 + 8 * a8 * t**7 + 7 * a7 * t**6 + 6 * a6 * t**5 + 5 * a5 * t**4 + 4 * a4 * t**3 + 3 * a3 * t**2 + 2 * a2 * t + a1
		theta_ddot 	= 72 * a9 * t**7 + 56 * a8 * t**6 + 42 * a7 * t**5 + 30 * a6 * t**4 + 20 * a5 * t**3 + 12 * a4 * t**2 + 6 * a3 * t + 2 * a2
		theta_dddot = 504 * a9 * t**6 + 336 * a8 * t**5 + 210 * a7 * t**4 + 120 * a6 * t**3 + 60 * a5 * t**2 + 24 * a4 * t + 6 * a3

		return t, theta, theta_dot, theta_ddot, theta_dddot

	def mltp_polynomial11th_4point(self):

		if len(self.path) != 4:
			print("The length of the path is not equal to four")
			return None

		theta0 = self.path[0]
		theta1 = self.path[1]
		theta2 = self.path[2]
		theta3 = self.path[3]
		t      = np.linspace(0, 1, self.sampling_number)

		a0 	= theta0
		a1 	= 0
		a2 	= 0
		a3 	= 0
		a4 	= 0
		a5 	= 3690.5625 * theta1 - 3014.71875 * theta0 - 1845.28125 * theta2 + 1169.4375 * theta3
		a6 	= 18795.75 * theta0 - 23988.65625 * theta1 + 14762.25 * theta2 - 9569.34375 * theta3
		a7 	= 64584.84375 * theta1 - 49087.96875 * theta0 - 46132.03125 * theta2 + 30635.15625 * theta3
		a8 	= 68523.75 * theta0 - 92264.0625 * theta1 + 73811.25 * theta2 - 50070.9375 * theta3
		a9 	= 73811.25 * theta1 - 53835.15625 * theta0 - 64584.84375 * theta2 + 44608.75 * theta3
		a10 = 22549.5 * theta0 - 31369.78125 * theta1 + 29524.5 * theta2 - 20704.21875 * theta3
		a11 = 5535.84375 * theta1 - 3932.15625 * theta0 - 5535.84375 * theta2 + 3932.15625 * theta3

		theta      	= a11 * t**11 + a10 * t**10 + a9 * t**9 + a8 * t**8 + a7 * t**7 + a6 * t**6 + a5 * t**5 + a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0
		theta_dot  	= 11 * a11 * t**10 + 10 * a10 * t**9 + 9 * a9 * t**8 + 8 * a8 * t**7 + 7 * a7 * t**6 + 6 * a6 * t**5 + 5 * a5 * t**4 + 4 * a4 * t**3 + 3 * a3 * t**2 + 2 * a2 * t + a1
		theta_ddot 	= 110 * a11 * t**9 + 90 * a10 * t**8 + 72 * a9 * t**7 + 56 * a8 * t**6 + 42 * a7 * t**5 + 30 * a6 * t**4 + 20 * a5 * t**3 + 12 * a4 * t**2 + 6 * a3 * t + 2 * a2
		theta_dddot = 990 * a11 * t**8 + 720 * a10 * t**7 + 504 * a9 * t**6 + 336 * a8 * t**5 + 210 * a7 * t**4 + 120 * a6 * t**3 + 60 * a5 * t**2 + 24 * a4 * t + 6 * a3

		return t, theta, theta_dot, theta_ddot, theta_dddot

	def mltp_cubicspline(self):
		'''
		Generates a trajectory based on Cubic Spline method for the path
		'''

		# calculate the velocity vector and the coeff matrix
		velocity_vector = self._cubicspline_velocity()
		coeff_matrix 	= self._cubicspline_coeff(velocity_vector)
		# time corresponding to position/velocity trajectory samples
		t 				= np.linspace(0, 1, self.sampling_number)
		# build time corresponding to each of the polynomials based on t - t_k
		t_interval 		= np.copy(t)

		# fix the time intervals
		counter = 0
		for i, e in enumerate(t):
			t_interval[i] 	= e - self.t_path[counter]
			next_step 		= (e >= self.t_path[counter+1]) and ((counter+1) < len(self.t_path))
			if next_step:
				counter += 1

		# initialize theta(t)
		theta 			= np.zeros_like(t)
		theta_dot 		= np.zeros_like(t)
		theta_ddot		= np.zeros_like(t)
		theta_dddot		= np.zeros_like(t)

		# Constructing the theta, theta', theta'', theta'''
		counter = 0
		for i, e in enumerate(t):

			theta[i] 			= 	coeff_matrix[counter][0] + \
									coeff_matrix[counter][1]*t_interval[i] + \
									coeff_matrix[counter][2]*t_interval[i]**2 + \
									coeff_matrix[counter][3]*t_interval[i]**3

			theta_dot[i] 		= 	coeff_matrix[counter][1] + \
									2*coeff_matrix[counter][2]*t_interval[i] + \
									3*coeff_matrix[counter][3]*t_interval[i]**2

			theta_ddot[i] 		= 	2*coeff_matrix[counter][2] + \
									6*coeff_matrix[counter][3]*t_interval[i]

			theta_dddot[i] 		= 	6*coeff_matrix[counter][3]

			if (e >= self.t_path[counter+1]) and ((counter+1) < len(self.t_path)):
				counter += 1

		return (t, theta, theta_dot, theta_ddot, theta_dddot)

	def _cubicspline_velocity(self):
		'''
		Helper function for calculating the velocity profile
		in cubic spline method
		'''
		v_i 			= 0
		v_f 			= 0
		A_prime 		= np.zeros((self.n-1, self.n-1))
		c_prime 		= np.zeros((self.n-1))
		velocity_vector = np.zeros_like(self.path)

		# making the A_prime matrix
		for i in range(self.n-1):
			A_prime[i, i] 			= 2*(self.T_path[i] + self.T_path[i+1])

			if 		i != 0:
				A_prime[i, i-1] 	= self.T_path[i+1]
			if 		i != self.n-2:
				A_prime[i, i+1] 	= self.T_path[i]

		# making the C_prime matrix
		for i in range(self.n-1):
			c_prime[i] 		= 3/(self.T_path[i]*self.T_path[i+1])*(self.T_path[i]**2*(self.path[i+2] -
							self.path[i+1]) + self.T_path[i+1]**2*(self.path[i+1] - self.path[i]))

			if 		i == 0:
				c_prime[i] -= self.T_path[i+1]*v_i
			elif 	i == self.n-2:
				c_prime[i] -= self.T_path[i]*v_f

		# calculating v vector from A_prime and C_prime matrices
		M = np.linalg.inv(A_prime)
		N = c_prime
		v = np.matmul(M, N)

		velocity_vector[0] 		= v_i
		velocity_vector[-1] 	= v_f
		velocity_vector[1:-1] 	= v

		return velocity_vector

	def _cubicspline_coeff(self, velocity_vector):
		'''
		Helper function for calculating the coefficient matrix
		in cubic spline method

		in the coefficient matrix we have:

		dimension 0 = number of coefficients in each polynomial
		dimension 1 = number of polynomials
		'''
		coeff = np.zeros((self.n, 4))

		coeff[:, 0] = self.path[0:self.n]
		coeff[:, 1] = velocity_vector[0:self.n]
		coeff[:, 2] = 1/self.T_path*(3*(self.path[1:self.n+1] - self.path[0:self.n])/self.T_path -
						2*velocity_vector[0:self.n] - velocity_vector[1:self.n+1])
		coeff[:, 3] = 1/self.T_path**2*(2*(- self.path[1:self.n+1] + self.path[0:self.n])/self.T_path +
						velocity_vector[0:self.n] + velocity_vector[1:self.n+1])

		return coeff

	def plot(self, results, method_name, _format='.pdf', _file_path='./results - multi point/'):
		_plot_trajectory(results, "mltp - " + method_name, _format, _file_path, labelsize=28)


# =================================================================================================
# -- shared plotting helper -----------------------------------------------------------------------
# =================================================================================================

def _plot_trajectory(results, method_name, _format, _file_path, labelsize):
	if not os.path.exists(_file_path):
		os.makedirs(_file_path)

	(t, theta, theta_dot, theta_ddot, theta_dddot) = results

	fig = plt.figure()
	fig.set_figheight(15)
	fig.set_figwidth(10)

	# Set global font to Times New Roman
	plt.rc('font', family='Times New Roman')

	has_jerk 	= theta_dddot is not None
	n_rows 		= 4 if has_jerk else 3

	# plot theta
	plt.subplot(n_rows, 1, 1)
	plt.plot(t, theta, linewidth=4)
	plt.xlabel(r"$t$", fontsize=28)
	plt.ylabel(r'$\theta$', fontsize=28)
	plt.tick_params(axis='both', which='major', labelsize=labelsize)

	# plot theta dot
	plt.subplot(n_rows, 1, 2)
	plt.plot(t, theta_dot, linewidth=4)
	plt.xlabel(r"$t$", fontsize=28)
	plt.ylabel(r'$\dot{\theta}$', fontsize=28)
	plt.tick_params(axis='both', which='major', labelsize=labelsize)

	# plot theta double dot
	plt.subplot(n_rows, 1, 3)
	plt.plot(t, theta_ddot, linewidth=4)
	plt.xlabel(r"$t$", fontsize=28)
	plt.ylabel(r'$\ddot{\theta}$', fontsize=28)
	plt.tick_params(axis='both', which='major', labelsize=labelsize)

	# plot theta triple dot
	if has_jerk:
		plt.subplot(n_rows, 1, 4)
		plt.plot(t, theta_dddot, linewidth=4)
		plt.xlabel(r"$t$", fontsize=28)
		plt.ylabel(r'$\dddot{\theta}$', fontsize=28)
		plt.tick_params(axis='both', which='major', labelsize=labelsize)

	plt.tight_layout()
	plt.savefig(_file_path + method_name + _format)
	plt.clf()


# =================================================================================================
# -- main -----------------------------------------------------------------------------------------
# =================================================================================================

if __name__ == "__main__":

	# ----- point to point -----
	path_planner = PathPlannerPTP(0, 1)

	path_planner.plot(path_planner.ptp_polynomial5th(),  "5th order polynomial")
	path_planner.plot(path_planner.ptp_polynomial7th(),  "7th order polynomial")
	path_planner.plot(path_planner.ptp_polynomial9th(),  "9th order polynomial")
	path_planner.plot(path_planner.ptp_bangbang(),       "Parabolic Method")
	path_planner.plot(path_planner.ptp_trapezoidal(),    "Trapezoidal Velocity Profile")
	path_planner.plot(path_planner.ptp_scurve(),         "S-curve Profile")

	# ----- multi point -----
	PATH = [0, 1, -1, 0]
	mltp_planner = PathPlannerMLTP(PATH)

	mltp_planner.plot(mltp_planner.mltp_ptpmethods("ptp_polynomial5th"), "ptp polynomial5th")
	mltp_planner.plot(mltp_planner.mltp_ptpmethods("ptp_polynomial7th"), "ptp polynomial7th")
	mltp_planner.plot(mltp_planner.mltp_ptpmethods("ptp_polynomial9th"), "ptp polynomial9th")
	mltp_planner.plot(mltp_planner.mltp_ptpmethods("ptp_bangbang"),      "ptp bangbang")
	mltp_planner.plot(mltp_planner.mltp_ptpmethods("ptp_trapezoidal"),   "ptp trapezoidal")
	mltp_planner.plot(mltp_planner.mltp_ptpmethods("ptp_scurve"),        "ptp scurve")

	mltp_planner.plot(mltp_planner.mltp_polynomial7th_4point(),  "7th order polynomial")
	mltp_planner.plot(mltp_planner.mltp_polynomial9th_4point(),  "9th order polynomial")
	mltp_planner.plot(mltp_planner.mltp_polynomial11th_4point(), "11th order polynomial")

	mltp_planner.plot(mltp_planner.mltp_cubicspline(), "cubic spline")
