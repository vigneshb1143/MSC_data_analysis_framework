import numpy as np

class orientTransform:
	"""
	Holds the orientation of a grain represented by euler angles in Bunge convention
	in terms of rotation matrix 
	Attributes:
	----------
	euler_angles : array like

	R : numpy array

	Methods:
	-------
	get_R(self, convert_to_rad=True)
	computes rotation matrix by elemental rotations Z1, X and Z2 in sequence
	
	get_local_coord(self, glob_vect)
	converts the vector represented in lab coordinates to material coordinates 

	get_glob_coord(self, local_vect)
	converts the vector represented in material coordinates to lab coordinates
	"""
	def __init__(self, euler_angles, convert_to_rad=True):
		self.get_R(Eu_angles=euler_angles, convert_to_rad=convert_to_rad)

	def get_R(self, Eu_angles, convert_to_rad=True):
		"""
		computes the rotation matrix required to move from lab to material coordinates
		convert_to_rad : Boolean 
		True if Euler angles supplied in degrees
		False if Euler angles are in radians
		"""
		if convert_to_rad:
			Eu_angles = [(angle_in_deg * np.pi) / 180.0 for angle_in_deg in Eu_angles]
		Z1 = np.array([[np.cos(Eu_angles[0]),np.sin(Eu_angles[0]),0], 
			[-np.sin(Eu_angles[0]),np.cos(Eu_angles[0]),0], [0,0,1]], dtype=float)
		X = np.array([[1,0,0], [0,np.cos(Eu_angles[1]),np.sin(Eu_angles[1])], 
			[0,-np.sin(Eu_angles[1]),np.cos(Eu_angles[1])]], dtype=float)
		Z2 = np.array([[np.cos(Eu_angles[2]),np.sin(Eu_angles[2]),0], 
			[-np.sin(Eu_angles[2]),np.cos(Eu_angles[2]),0], [0,0,1]], dtype=float)
		self.R = Z2 @ X @ Z1
		return 

	def get_local_coord(self, glob_vect):
		"""
		converts the vector represented in lab coordinates to material coordinates
		glob_vect : numpy array
		"""
		local_vect = self.R @ glob_vect
		return local_vect

	def get_glob_coord(self, local_vect):
		"""
		converts the vector represented in material coordinates to lab coordinates
		local_vect : numpy array
		"""
		R_inv = np.linalg.inv(self.R)
		glob_vect = R_inv @ local_vect
		return glob_vect


if __name__ == "__main__":
	orientation = orientTransform([0.0, 0.0, 90.0], convert_to_rad=True)
	print(orientation.get_local_coord(np.array([0,0,1])))
	print(orientation.get_local_coord(np.array([1,0,0])))
	print(orientation.get_glob_coord(np.array([1,0,0])))
