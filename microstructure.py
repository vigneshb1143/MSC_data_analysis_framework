import numpy as np
from crystal_functions import elastic_modulus_hkl, get_schmid_table, get_ipf_color_orix, eu2qu, fcc_slip_systems
from orientation_transformations import orientTransform
from orix.quaternion import Misorientation
from orix.quaternion.symmetry import get_point_group
from utils import make_unit_vect

class microstructure:
	"""
	Author: Vignesh Babu Rao
	This class holds grain ids and the corresponding orientation (Euler angles)
	by reading losalamosFFT file written by Dream3D
	Attributes:
	----------
	micro : numpy array
	format: an array of [Phi1   Phi   Phi2  X  Y   Z  Feature_ID   Phase_ID]

	Methods:
	-------
	euler_table(self)
	return euler table

	quaternion_table(self)
	return quaternion table

	EM_table(self)
	return Elastic modulus table
	"""
	def __init__(self, euler_file):
		print('\nReading Euler file')
		temp = np.genfromtxt(euler_file)
		print('Done reading Euler file')
		_, unique_idx, unique_counts = np.unique(temp[:,6], return_index=True, return_counts=True)
		micro = temp[unique_idx]
		micro = np.concatenate((micro, unique_counts.reshape(-1,1)), axis=1)
		# idx = np.argsort(micro[:,6])
		self.micro = micro #[:,8+1]
		self.euler_table = self.euler_table()
		self.quaternion_table = self.quaternion_table()
		self.EM_table = self.EM_table()
		self.schmid_table, self.schmid_plane_sum_table, self.schmid_plane_max_table = self.schmid_table()
		self.grain_size_table = self.get_grain_size_table()
		self.misorientation_matrix = []
		self.loc_coord_table = []
		self.slip_plane_normals_table_loc = []
		self.slip_plane_normals_table_glob = []
	
	def euler_table(self):
		"""
		creates euler angles table that has [grain_id, Phi1, Phi, Phi2] 
		"""
		euler_table = []
		for grain in self.micro:
			euler_table.append([int(grain[6]), grain[0], grain[1], grain[2]])
		return np.array(euler_table)

	def quaternion_table(self):
		"""
		creates quaternions table that has [grain_id, q0, q1, q2, q3] 
		"""
		micro = self.micro
		quaternion = []
		P = 1.0
		micro[:,0] = micro[:,0] * np.pi / 180.0
		micro[:,1] = micro[:,1] * np.pi / 180.0
		micro[:,2] = micro[:,2] * np.pi / 180.0
		sigma = (micro[:,0] + micro[:,2]) / 2
		delta = (micro[:,0] - micro[:,2]) / 2
		c = np.cos(micro[:,1] / 2)
		s = np.sin(micro[:,1] / 2)
		quaternion.append(micro[:,6].astype(int))
		quaternion.append(c * np.cos(sigma))
		quaternion.append(-1.0 * P * s * np.cos(delta))
		quaternion.append(-1.0 * P * s * np.sin(delta))
		quaternion.append(-1.0 * P * c * np.sin(sigma))
		quaternion = np.array(quaternion).T
		neg_idx = np.where(quaternion[:,1]<0)
	#     print(neg_idx)
		quaternion[neg_idx, 1:] = -1 * quaternion[neg_idx, 1:]
		return quaternion

	def EM_table(self):
		"""
		computes elastic modulus along a specific [hkl] plane normal direction
		creates elastic modulus table [grain_id, E_hkl]
		"""
		EM_table = []
		for grain in self.micro:
			EM_table.append([int(grain[6]), elastic_modulus_hkl(grain[0:3])])
		return np.array(EM_table)

	def schmid_table(self):
		"""get_schmid_table:
			calculates the schmid factor for all the slip systems and sorts them based on value
			return: numpy array of shape(N_grains, 1+12)"""
		return get_schmid_table(self.euler_table, return_plane_sum=True, return_plane_max=True)

	def get_grain_size_table(self):
		"""calculates the number of voxel corresponding to each grain as well as its volume fraction
			return: numpy array of shape(N_grains, 1+2)
					[gid, num_voxels, vol_fraction]"""
		return np.concatenate((self.micro[:,6].reshape(-1,1), self.micro[:,8].reshape(-1,1), \
			(self.micro[:,8]/np.sum(self.micro[:,8])).reshape(-1,1)), axis=1)


	def get_misorientation_matrix(self):
		"""returns the symmetry reduced smallest angle of rotation transforming every 
		grain orientation to every other orientation using get_distance_matrix in orix
		"""
		sym_al = get_point_group(225)
		mori = Misorientation(self.quaternion_table[:,1:5], symmetry=(sym_al, sym_al))
		self.misorientation_matrix = mori.get_distance_matrix(progressbar=False, degrees=True)
		return

	def get_slip_plane_normals_table(self, coord='local'):
		"""returns the four slip plane normals for each grain either in material coordinates or global coordinates
		return: numpy array[gid, slip plane1 normal X, slip plane1 normal Y, slip plane1 normal Z, ...]"""
		slip_planes = fcc_slip_systems(opt='slip_plane')
		unit_slip_planes = [make_unit_vect(each) for each in slip_planes]
		if coord == 'local':
			unit_slip_planes_arr = np.tile(np.array(unit_slip_planes).ravel(), (len(self.euler_table), 1))
			slip_plane_normals_table = np.concatenate((self.euler_table[:,0].reshape(-1,1), unit_slip_planes_arr), axis=1) 
			self.slip_plane_normals_table_loc = slip_plane_normals_table
			return
		elif coord == 'global':
			slip_plane_normals_table = []
			for g in self.euler_table:
				R = orientTransform(g[1:4])
				temp = [g[0].item()]
				for sp in unit_slip_planes:
					temp.extend(R.get_glob_coord(sp))
				slip_plane_normals_table.append(temp)
			self.slip_plane_normals_table_glob = np.array(slip_plane_normals_table)
			return
		else:
			raise NotImplementedError('invalid coordinate system')

	def get_local_coordinates_table(self, ref_dir=[0,0,1]):
		"""
		returns how a given ref_dir is represented in material coordinates 
		of each of the grains in the Euler table"""
		loc_coord_table = []
		for each in self.euler_table:
			R = orientTransform(each[1:4])
			loc_coord_table.append(R.get_local_coord(ref_dir))
		loc_coord_table = np.concatenate((self.euler_table[:,0].reshape(-1,1), np.array(loc_coord_table)), axis=1)
		self.loc_coord_table = loc_coord_table
		return

	def get_ipf_color_table(self, ref_dir):
		"""returns ipfcolor table [grain_ID, r, g, b] """
		ipf_color_table = []
		for eu in self.euler_table:
			ipf_color_table.append(np.append(eu[0], get_ipf_color_orix(eu[1:4], ref_dir)))
		return np.array(ipf_color_table)


if __name__ == "__main__":
	micr = microstructure('/home/vignesh/projects/CP_simulation/AI1/S00/06-Simulation/Euler_V100100100.txt')
	eu_t = micr.euler_table
	q_t = micr.quaternion_table
	E_T = micr.EM_table
	s_T = micr.schmid_table
	ipf_table = micr.get_ipf_color_table(ref_dir=[0,0,1])
	print(eu_t.shape, q_t.shape, E_T.shape, s_T.shape, ipf_table.shape)
	# print(ipf_table)
	# print('Printing inconsistencies')
	# for ipf_col in ipf_table:
	# 	if (ipf_col[1:] > 1.0).any():
	# 		print(ipf_col)
	q =eu2qu(np.array([0, 54.74, 45.0]).reshape((1,3)))
	# print(q)
	# print('testing ipf colors: ', get_ipf_color(q[0], [0,0,-1]))

	#testing schmid table
	# print('Schmid table: \n')
	# for m in s_T:
	# 	print(m)
	micr.get_misorientation_matrix()
	print('Misorientation:')
	print('min_mori: ', np.min(micr.misorientation_matrix), ' max_mori: ', np.max(micr.misorientation_matrix))
	print(micr.misorientation_matrix)

	#testing grain_size_table
	print('grain size table:')
	print(micr.grain_size_table)
	print('total voxels: ', np.sum(micr.grain_size_table[:,1]), 'total volume fraction: ', np.sum(micr.grain_size_table[:,2]))

	#testing slip_plane_normals_table
	tmp1 =  micr.get_slip_plane_normals_table(coord='local')
	print('slip plane normals table local shape:', tmp1.shape)
	print('slip plane normals table local: \n', tmp1)
	tmp2 = micr.get_slip_plane_normals_table(coord='global')
	print('slip plane normals table global shape:', tmp2.shape)
	print('slip plane normals table global: \n', tmp2)

