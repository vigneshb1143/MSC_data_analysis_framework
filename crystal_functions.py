#Author: Vignesh Babu Rao 
#Year: 2022
import numpy as np
from orientation_transformations import orientTransform
from utils import find_angle, quaternionXvector, cart2sph
import matplotlib.pyplot as plt
from orix.quaternion import symmetry, Orientation
from orix import plot
from orix.vector import Vector3d

cubic_plane_family = {'100':[[0,0,1], [0,1,0], [1,0,0], [0,0,-1], [0,-1,0], [-1,0,0]], \
'110':[[1,1,0], [1,-1,0], [1,0,1], [1,0,-1], [0,1,1], [0,-1,1], [-1,-1,0], [-1,1,0], [-1,0,-1], [-1,0,1], [0,-1,-1], [0,1,-1]], \
'111':[[1,1,1], [-1,1,1], [1,-1,1], [1,1,-1], [-1,-1,-1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]}

def fcc_slip_systems(opt='slip_system'):
	if opt == 'slip_system':
		slip_sys_list = []
		slip_sys_list = [slip_sys_list.append([]) for i in range(12)]
		slip_sys_list[0] = [[1, 1, 1], [1, -1, 0]]
		slip_sys_list[1] = [[1, 1, 1], [-1, 0, 1]]
		slip_sys_list[2] = [[1, 1, 1], [0, 1, -1]]
		slip_sys_list[3] = [[-1, 1, 1], [1, 0, 1]]
		slip_sys_list[4] = [[-1, 1, 1], [-1, -1, 0]]
		slip_sys_list[5] = [[-1, 1, 1], [0, 1, -1]]
		slip_sys_list[6] = [[1, -1, 1], [-1, 0, 1]]
		slip_sys_list[7] = [[1, -1, 1], [0, -1, -1]]
		slip_sys_list[8] = [[1, -1, 1], [1, 1, 0]]
		slip_sys_list[9] = [[-1, -1, 1], [-1, 1, 0]]
		slip_sys_list[10] = [[-1, -1, 1], [1, 0, 1]]
		slip_sys_list[11] = [[-1, -1, 1], [0, -1, -1]]
		return slip_sys_list
	elif opt == 'slip_plane':
		return [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]]

	else:
		raise Exception('wrong option')

def get_euler_angle(surface_plane_normal='001'):
	if surface_plane_normal == '001':
		return [0.0, 0.0, 0.0]
	elif surface_plane_normal == '101':
		return [270.0, 45.0, 90.0]
	elif surface_plane_normal == '111':
		return [0, 54.74, 45.0]
	else:
		raise NotImplementedError('Not implemented')

def elastic_modulus_hkl(Eu_angle):
	C11 = 107300.0 #MPa This is AlMgSi used in your CP work
	C12 = 60800.0 #MPa
	C44 = 28300.0 #MPa
	S11 = (C11 + C12) / ((C11 - C12) * (C11 + 2 * C12))
	S12 = -C12 / ((C11 - C12) * (C11 + 2 * C12))
	S44 = 1 / C44

	ori = orientTransform(Eu_angle, convert_to_rad=True)
	hkl = ori.get_local_coord([0, 0, 1])

	alpha = np.dot(hkl, [1, 0, 0]) / \
	(np.linalg.norm(hkl) * np.linalg.norm([1, 0, 0]))
	beta = np.dot(hkl, [0, 1, 0]) / \
	(np.linalg.norm(hkl) * np.linalg.norm([0, 1, 0]))
	gamma = np.dot(hkl, [0, 0, 1]) / \
	(np.linalg.norm(hkl) * np.linalg.norm([0, 0, 1]))
	E_hkl = 1 / (S11 - 2 * (S11 - S12 - 0.5 * S44) * \
		(alpha**2 * beta**2 + beta**2 * gamma**2 + alpha**2 * gamma**2))
	return E_hkl

def get_schmid_table(euler_table, return_plane_sum=False, return_plane_max=False):
	"""get_schmid_table:
		calculates the schmid factor for all the slip systems and sorts them based on value
		return: numpy array of shape(N_grains, 13)"""
	
	slip_sys_list = fcc_slip_systems()
	schmid_table = []
	schmid_plane_sum_table = []
	schmid_plane_max_table = []
	for eu in euler_table:
		# phi_list = []
		# lamda_list = []
		schmid_factor_list = []
		for slip_sys in slip_sys_list:
			slip_plane_normal = slip_sys[0]
			slip_dir = slip_sys[1]
			# R = get_R(eu[1:4], convert_to_rad=True)
			ori = orientTransform(eu[1:4], convert_to_rad=True)
			loading_dir = ori.get_local_coord([0.0, 0.0, 1.0])
#             print('(slip_plane_normal, slip_dir, loading_dir):', slip_plane_normal, slip_dir, loading_dir)
			phi = find_angle(loading_dir, slip_plane_normal)
			lamda = find_angle(loading_dir, slip_dir)
			if phi > np.pi/2:
				phi = np.pi - phi
			if lamda > np.pi/2:
				lamda = np.pi - lamda
			schmid_factor = np.cos(phi) * np.cos(lamda)
			schmid_factor_list.append(round(schmid_factor, 5))
		if return_plane_sum == True:
			schmid_plane_sum_table.append([eu[0].item(), sum(schmid_factor_list[0:3]), sum(schmid_factor_list[3:6]), \
				sum(schmid_factor_list[6:9]), sum(schmid_factor_list[9:12])])
		if return_plane_max == True:
			schmid_plane_max_table.append([eu[0].item(), max(schmid_factor_list[0:3]), max(schmid_factor_list[3:6]), \
				max(schmid_factor_list[6:9]), max(schmid_factor_list[9:12])])
		schmid_table.append([eu[0]]+sorted(schmid_factor_list, reverse=True))

	if return_plane_max == True and return_plane_sum == True:
		return np.array(schmid_table), np.array(schmid_plane_sum_table), np.array(schmid_plane_max_table)
	elif return_plane_sum == True and return_plane_max == False:
		return np.array(schmid_table), np.array(schmid_plane_sum_table)
	elif return_plane_sum == False and return_plane_max == True:
		return np.array(schmid_table), np.array(schmid_plane_max_table) 
	else:
		return np.array(schmid_table)

def eu2qu(euler):
	"""Returns the unit quaternions given Euler angles in degrees"""
	eulerr = euler.copy()
	quaternion = []
	P = 1.0
	eulerr[:,0] = euler[:,0] * np.pi / 180.0
	eulerr[:,1] = euler[:,1] * np.pi / 180.0
	eulerr[:,2] = euler[:,2] * np.pi / 180.0
	sigma = (eulerr[:,0] + eulerr[:,2]) / 2
	delta = (eulerr[:,0] - eulerr[:,2]) / 2
	c = np.cos(eulerr[:,1] / 2)
	s = np.sin(eulerr[:,1] / 2)
	quaternion.append(c * np.cos(sigma))
	quaternion.append(-1.0 * P * s * np.cos(delta))
	quaternion.append(-1.0 * P * s * np.sin(delta))
	quaternion.append(-1.0 * P * c * np.sin(sigma))
	quaternion = np.array(quaternion).T
	print('quat: ', quaternion)
	neg_idx = np.where(quaternion[:,0]<0)
	quaternion[neg_idx, 0:] = -1 * quaternion[neg_idx, 0:] #to force quaternions to lie in Northern hemisphere
	return quaternion

# def get_ipf_color(quaternion, reference_direction): #buggy code. stop using this
# 	"""Map vectors to a color
# 	vectors supplied should be unit vectors
# 	quaternions: in orix, rotations are represented as unit quaternions
# 	reference_direction: this can be one single vector or an array of vectors whose 
# 	size is same as the number of quaternions"""
# 	# adobpted from source: https://githubmemory.com/repo/pyxem/orix/issues/166
# 	# the columns of this matrix should map to red, green, blue respectively
# 	try:
# 		assert(abs(np.linalg.norm(quaternion) - 1.0) < 1e-5)
# 	except:
# 		raise Exception('Given quaternion is not a UNIT quaternion!')
# 	try:
# 		assert(abs(np.linalg.norm(reference_direction) - 1.0) < 1e-5)
# 	except:
# 		raise Exception('Given reference direction does not represent a UNIT vector')

# 	color_corners = np.array([[0, 1/np.sqrt(2), 1/np.sqrt(3)],
# 							  [0, 0, 1/np.sqrt(3)],
# 							  [1, 1/np.sqrt(2), 1/np.sqrt(3)]])
# 	color_mapper = np.linalg.inv(color_corners)

# 	# a bit of wrangling
# 	data_sol = quaternionXvector(quaternion, reference_direction)
# 	# quaternionXvector returns a vector as a numpy array
# 	data_sol = data_sol / np.linalg.norm(data_sol)
# 	# print(np.product(data_sol.shape[:-1]))
# 	flattened = data_sol.reshape(np.product(data_sol.shape[:-1]), 3).T
# 	# print('flattened data: ', flattened, '\n', 'its shape: ', flattened.shape)
# 	rgb_mapped = np.dot(color_mapper, flattened)
# 	# print('dot product: ', rgb_mapped, '\n its shape: ', rgb_mapped.shape)
# 	rgb_mapped = np.abs(rgb_mapped / np.abs(rgb_mapped).max(axis=0)).T
# 	# print('scaled rgb: ', rgb_mapped, '\n its shape: ', rgb_mapped.shape)
# 	rgb_mapped = rgb_mapped.reshape(data_sol.shape)**0.5
# 	# print('final rgb: ', rgb_mapped, '\n its shape: ', rgb_mapped.shape)
# 	return rgb_mapped

def get_ipf_color_orix(eu, n):
	sym_al = symmetry.get_point_group(225)
	a = Orientation.from_euler(eu, degrees=True, symmetry=sym_al)
	ipfkey = plot.IPFColorKeyTSL(sym_al,direction=Vector3d(n)) #n should be in sample direction. Orix transforms it to crystal coordinates.
	rgb = ipfkey.orientation2color(a)
	return rgb 

# sample <---> crystal coordinate system transformation using Euler angles
def get_R(Eu_angles, convert_to_rad=True):
	if convert_to_rad:
		# print('Converting Euler angles from Deg to Rad')
		Eu_angles = [(angle_in_deg * np.pi) / 180.0 for angle_in_deg in Eu_angles]
#         print(Eu_angles)
	Z1 = np.array([[np.cos(Eu_angles[0]),np.sin(Eu_angles[0]),0], [-np.sin(Eu_angles[0]),np.cos(Eu_angles[0]),0], [0,0,1]], dtype=float)
	X = np.array([[1,0,0], [0,np.cos(Eu_angles[1]),np.sin(Eu_angles[1])], [0,-np.sin(Eu_angles[1]),np.cos(Eu_angles[1])]], dtype=float)
	Z2 = np.array([[np.cos(Eu_angles[2]),np.sin(Eu_angles[2]),0], [-np.sin(Eu_angles[2]),np.cos(Eu_angles[2]),0], [0,0,1]], dtype=float)
	R = Z2 @ X @ Z1
	return R

#get vector components in crystal coordinates 
def get_local_coord(glob_vect, R): 
	local_vect = R @ glob_vect
	return local_vect

#get vector components in sample coordinates
def get_glob_coord(local_vect, R):
	R_inv = np.linalg.inv(R)
	glob_vect = R_inv @ local_vect
	return glob_vect


def plot_pole_figure(xy, P001, P110, P111, colors=[], save_fig='none', annotate_poles=False, title='none'):
	"""
	xy is the steriographic projections of all the poles you want to plot
	P001 are the steriographic projections of family of poles {001, 010, 100, 00-1, 0-10, -100}
	P110 are the steriographic projections of family of poles 
	{110, 1-10, 101, 10-1, 011, 0-11, -1-10, -110, -10-1, -101, 0-1-1, 01-1}
	P111 are the steriographic projections of family of poles {111, -111, 1-11, 11-1, -1-1-1, 1-1-1, -11-1, -1-11}
	all arguments has to be passed as numpy arrays"""
	if len(colors) == 0:
		colors = np.zeros((xy.shape[0],3))

	a = plt.figure()
	ax = a.gca()
	theta = np.linspace(0,2*np.pi,100)
	ax.plot(1*np.cos(theta), 1*np.sin(theta), color=[0,0,0])
	ax.scatter(P001[:,0], P001[:,1], color=[1,0,0], marker='*', s=100)
	ax.scatter(P110[:,0], P110[:,1], color=[0,1,0], marker='*', s=100)
	ax.scatter(P111[:,0], P111[:,1], color=[0,0,1], marker='*', s=100)
	ax.scatter(xy[:,0], xy[:,1], color=colors, alpha=0.75)
	if annotate_poles == True:
		text = [','.join([str(cubic_plane_family['100'][i][j]) for j in range(3)]) for i in range(len(cubic_plane_family['100']))]
		for i, txt in enumerate(text):
			if i >= len(text)/2:
				ax.annotate(txt, xy=(P001[i,0],P001[i,1]+0.05), rotation=90)
			else:
				ax.annotate(txt, xy=(P001[i,0]+0.1,P001[i,1]+0.05), rotation=90)
		text = [','.join([str(cubic_plane_family['110'][i][j]) for j in range(3)]) for i in range(len(cubic_plane_family['110']))]
		for i, txt in enumerate(text):
			if i >= len(text)/2:
				ax.annotate(txt, xy=(P110[i,0],P110[i,1]+0.05), rotation=90)
			else:
				ax.annotate(txt, xy=(P110[i,0]+0.1,P110[i,1]+0.05), rotation=90)
		text = [','.join([str(cubic_plane_family['111'][i][j]) for j in range(3)]) for i in range(len(cubic_plane_family['111']))]
		for i, txt in enumerate(text):
			if i >= len(text)/2:
				ax.annotate(txt, xy=(P111[i,0],P111[i,1]+0.05), rotation=90)
			else:
				ax.annotate(txt, xy=(P111[i,0]+0.1,P111[i,1]+0.05), rotation=90)
	ax.set_xlim([-1,1])
	ax.set_ylim([-1,1])
	ax.axis('equal')
	ax.tick_params(axis='both', which='major', labelsize=12)
	ax.set_xlabel('x-axis', fontsize=16)
	ax.set_ylabel('y-axis', fontsize=16)
	if title != 'none':
		ax.set_title(title, fontsize=18)
	if save_fig != 'none': 
		plt.savefig(save_fig, dpi=300)
	plt.show()
	return