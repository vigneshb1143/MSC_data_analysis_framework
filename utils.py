import numpy as np

def cart2sph(xyz):
    sph = np.zeros(xyz.shape) #rho, theta, phi
    sph[:,0] = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    sph[:,1] = np.arccos(xyz[:,2] / sph[:,0])
    sph[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return sph

def stereographic_proj(sph):
	stereo_proj = np.zeros((sph.shape[0],2))
	stereo_proj[:,0] = np.tan(sph[:,1]/2) * np.cos(sph[:,2])
	stereo_proj[:,1] = np.tan(sph[:,1]/2) * np.sin(sph[:,2])
	return stereo_proj

def find_angle(vect_A, vect_B):
	"""returns the angle between two vectors in radians
	Range of arccos is (0,pi). So you will always end up getting a positive angle"""
	return np.arccos(np.dot(vect_A, vect_B) / (np.linalg.norm(vect_A) * np.linalg.norm(vect_B)))

def find_unit_norm(three_points):
	"""Returns the unit normal vector of a plane (or triangle) given any three
	points on the plane"""
	p = three_points[0]
	q = three_points[1]
	r = three_points[2]
	pq = np.subtract(q,p)  #pq_vect = oq_vect - op_vect
	pr = np.subtract(r,p)  #pr_vect = or_vect - op_vect
	norm_vect = np.cross(pq,pr)
	unit_norm_vect = norm_vect / np.linalg.norm(norm_vect)
	return unit_norm_vect

def make_unit_vect(vect):
	vect = np.array(vect)
	unit_vect = vect/np.linalg.norm(vect)
	return unit_vect

def area_of_triangle(mtd, desc, projection=False, plane_normal=[0,0,0]):
	"""
	Returns the area of a triangle if projection=False
	Returns the [area, projected_area] of the triangle if projection=True
	mtd: 'coord'; other methods can be implemented in future if needed 
	desc: if mtd is 'coord', then description should be the coordinates (x,y,z)\
		  of three vertices of the triangle
	projection: if you want to find the area of triangle projected on to an\
				plane in addition to its true area, set this to True
	plane_normal: normal vector of the plane to which the triangle has to be\
				  projected. This is used only if projection is set to True  
	"""
	if mtd == 'coord':
		p = desc[0]
		q = desc[1]
		r = desc[2]
		pq = np.subtract(q,p)  #pq_vect = oq_vect - op_vect
		pr = np.subtract(r,p)  #pr_vect = or_vect - op_vect
		area = 0.5 * np.linalg.norm(np.cross(pq,pr))  # areaa = 0.5*||pq_vect x pr_vect||

		if projection == True:
			plane_normal = np.array(plane_normal)
			#finding pq vectors projection on the plane
			proj_pq = pq - np.dot(pq, plane_normal)/(np.linalg.norm(plane_normal)**2) * plane_normal
			#finding pr vectors projection on the plane
			proj_pr = pr - np.dot(pr, plane_normal)/(np.linalg.norm(plane_normal)**2) * plane_normal
			proj_area = 0.5 * np.linalg.norm(np.cross(proj_pq,proj_pr))  # areaa = 0.5*||pq_vect x pr_vect||
			return [area, proj_area]

		return area		
	else:
		raise NotImplementedError('The specified method of finding area is not implemented')

def rotate_vector(vect, angle, axis='z'):
	if axis == 'z':
		theta_z = angle
		z_rot = np.array([(np.cos(theta_z),-np.sin(theta_z),0.0), (np.sin(theta_z),np.cos(theta_z),0.0), np.array([0.0,0.0,1.0])])
		rot_vect = np.matmul(z_rot, vect)
	elif axis == 'x':
			raise NotImplementedError('grid rotation along x-axis is currently not implemented')
	elif axis == 'y':
			raise NotImplementedError('grid rotation along y-axis is currently not implemented')
	else:
		raise Exception('Invalid axis')
	return rot_vect

def get_radial_plane_norm_from_ang(angle):
	"""Returns the radial plane normals given the angular location of the crack front
		Very specific to high-fidelity framework
		angle: angular location of crack front in radians
		return: unit normal vector to the radial plane """
	vect = [1,0,0]
	plane_norm = make_unit_vect(rotate_vector(vect, angle, axis='z'))
	return plane_norm

def PCA(data, correlation = False, sort = True):
	""" Applies Principal Component Analysis to the data
	source: https://stackoverflow.com/questions/38754668/plane-fitting-in-a-3d-point-cloud
	Parameters
	----------        
	data: array
	    The array containing the data. The array must have NxM dimensions, where each
	    of the N rows represents a different individual record and each of the M columns
	    represents a different variable recorded for that individual record.
	        array([
	        [V11, ... , V1m],
	        ...,
	        [Vn1, ... , Vnm]])

	correlation(Optional) : bool
	        Set the type of matrix to be computed (see Notes):
	            If True compute the correlation matrix.
	            If False(Default) compute the covariance matrix. 
	            
	sort(Optional) : bool
	        Set the order that the eigenvalues/vectors will have
	            If True(Default) they will be sorted (from higher value to less).
	            If False they won't.   
	Returns
	-------
	eigenvalues: (1,M) array
	    The eigenvalues of the corresponding matrix.
	    
	eigenvector: (M,M) array
	    The eigenvectors of the corresponding matrix.

	Notes
	-----
	The correlation matrix is a better choice when there are different magnitudes
	representing the M variables. Use covariance matrix in other cases.

	"""

	mean = np.mean(data, axis=0)

	data_adjust = data - mean

	#: the data is transposed due to np.cov/corrcoef syntax
	if correlation:
	    
	    matrix = np.corrcoef(data_adjust.T)
	    
	else:
	    matrix = np.cov(data_adjust.T) 

	eigenvalues, eigenvectors = np.linalg.eig(matrix)

	if sort:
	    #: sort eigenvalues and eigenvectors
	    sort = eigenvalues.argsort()[::-1]
	    eigenvalues = eigenvalues[sort]
	    eigenvectors = eigenvectors[:,sort]

	return eigenvalues, eigenvectors

def best_fitting_plane(points, equation=False):
	""" Computes the best fitting plane of the given points

	Parameters
	----------        
	points: array
	    The x,y,z coordinates corresponding to the points from which we want
	    to define the best fitting plane. Expected format:
	        array([
	        [x1,y1,z1],
	        ...,
	        [xn,yn,zn]])
	        
	equation(Optional) : bool
	        Set the oputput plane format:
	            If True return the a,b,c,d coefficients of the plane.
	            If False(Default) return 1 Point and 1 Normal vector.  
	Ref:  https://stackoverflow.com/questions/38754668/plane-fitting-in-a-3d-point-cloud
	Returns
	-------
	a, b, c, d : float
	    The coefficients solving the plane equation.

	or

	point, normal: array
	    The plane defined by 1 Point and 1 Normal vector. With format:
	    array([Px,Py,Pz]), array([Nx,Ny,Nz])
	    
	"""

	w, v = PCA(points)

	#: the normal of the plane is the last eigenvector
	normal = v[:,2]
	   
	#: get a point from the plane
	point = np.mean(points, axis=0)


	if equation:
	    a, b, c = normal
	    d = -(np.dot(normal, point))
	    return a, b, c, d
	    
	else:
	    return point, normal 

def get_isoparametric(node_coords, point):
	x, y, z = point
	x1, y1, z1 = node_coords[0]
	x2, y2, z2 = node_coords[1]
	x3, y3, z3 = node_coords[2]
	x4, y4, z4 = node_coords[3]
	SV1 = x2 * (y3 * z4 - y4 * z3) + x3 * (y4 * z2 - y2 * z4) + x4 * (y2 * z3 - y3 * z2); #6V1
	SV2 = x1 * (y4 * z3 - y3 * z4) + x3 * (y1 * z4 - y4 * z1) + x4 * (y3 * z1 - y1 * z3); #6V2
	SV3 = x1 * (y2 * z4 - y4 * z2) + x2 * (y4 * z1 - y1 * z4)+ x4 * (y1 * z2 - y2 * z1); #6V3
	SV4 = x1 * (y3 * z2 - y2 * z3) + x2 * (y1 * z3 - y3 * z1)+ x3 * (y2 * z1 - y1 * z2); #6V4
	V = (SV1 + SV2 + SV3 + SV4) / 6;
	a1 = (y4-y2)*(z3-z2) - (y3-y2)*(z4-z2)
	b1 = (x3-x2)*(z4-z2) - (x4-x2)*(z3-z2)
	c1 = (x4-x2)*(y3-y2) - (x3-x2)*(y4-y2)
	xi1 = (SV1 + a1*x + b1*y + c1*z)/(6*V)
	a2 = (y3-y1)*(z4-z3) - (y3-y4)*(z1-z3)
	b2 = (x4-x3)*(z3-z1) - (x1-x3)*(z3-z4)
	c2 = (x3-x1)*(y4-y3) - (x3-x4)*(y1-y3)
	xi2 = (SV2 + a2*x + b2*y + c2*z)/(6*V)
	a3 = (y2-y4)*(z1-z4) - (y1-y4)*(z2-z4)
	b3 = (x1-x4)*(z2-z4) - (x2-x4)*(z1-z4)
	c3 = (x2-x4)*(y1-y4) - (x1-x4)*(y2-y4)
	xi3 = (SV3 + a3*x + b3*y + c3*z)/(6*V)
	a4 = (y1-y3)*(z2-z1) - (y1-y2)*(z3-z1)
	b4 = (x2-x1)*(z1-z3) - (x3-x1)*(z1-z2)
	c4 = (x1-x3)*(y2-y1) - (x1-x2)*(y3-y1)
	xi4 = (SV4 + a4*x + b4*y + c4*z)/(6*V)
	return xi1, xi2, xi3, xi4

def quaternionXvector(q, v): 
    """Quaternion-Vector multiplication"""
    # source: https://github.com/pyxem/orix/blob/master/orix/quaternion/quaternion.py
    a, b, c, d = q[0], q[1], q[2], q[3],
    x, y, z = v[0], v[1], v[2]
    x_new = (a ** 2 + b ** 2 - c ** 2 - d ** 2) * x + 2 * (
        (a * c + b * d) * z + (b * c - a * d) * y
    )
    y_new = (a ** 2 - b ** 2 + c ** 2 - d ** 2) * y + 2 * (
        (a * d + b * c) * x + (c * d - a * b) * z
    )
    z_new = (a ** 2 - b ** 2 - c ** 2 + d ** 2) * z + 2 * (
        (a * b + c * d) * y + (b * d - a * c) * x
    )
    new_v = np.array([x_new, y_new, z_new]).reshape(-1, 3)
    return new_v

def write_XDMF(file_name, time_steps, topology_dict, geometry_dict, attribute_dict):
	f = open(file_name, 'w')
	f.write('<?xml version="1.0" ?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n\
<Xdmf>\n    <Domain>\n   <Grid CollectionType="Temporal" \
		GridType="Collection" Name="TimeSeries">\n') 
	for step in time_steps:
		f.write(f'\n            <Grid Name="T{step}" > \n')

		#write topology
		f.write(f'\n                <Topology Type="Tetrahedron"\
 NumberOfElements="{topology_dict["num_elem"]}" >\n \
			        <DataItem Format="HDF" DataType="Float"\
 Dimensions="{topology_dict["dim"]} 4">')
		f.write(f'\n                        {topology_dict["basename"]}_MESH.h5:/EleConn\n')
		f.write(f'                    </DataItem>\n                </Topology>\n')
		
		#write geometry
		f.write(f'\n                <Geometry Type="XYZ">\n                    <DataItem Format="HDF" DataType="Float" \
 Precision="8" Dimensions="{geometry_dict["dim"]} 3>">\n')
		f.write(f'\n                        {geometry_dict["basename"]}_{step}.h5:/DefCoords\n')
		f.write(f'                    </DataItem>\n                </Geometry>\n')

		#write time value
		f.write(f'                <Time Value="{step}" />\n')
		#write attributes
		for name in attribute_dict["names"]:
			f.write(f'                <Attribute Name="{name}" Center="Node">\n\
					<DataItem Format="HDF" DataType="Float" Precision="8" Dimensions="{attribute_dict["dim"]} 1">')
			f.write(f'\n                        {attribute_dict["basename"]}_\
{step}.h5:/{name}\n                    </DataItem>\n\
				</Attribute>\n')
		f.write('\n            </Grid>')

	f.write('\n        </Grid>\n    </Domain>\n</Xdmf>')
	f.close()
	return








	# fs << "\n            <Grid Name=\"T" << incrementID << "\"> \n";
	# // Write topology and element connectivity
	# fs << "\n                <Topology Type=\"Tetrahedron\" NumberOfElements=\"" << numberOfElements << "\" >\n                    <DataItem Format=\"HDF\" DataType=\"Float\" Dimensions=\"" <<
	#   numberOfElements << " 4\">\n";
	# fs << "\n                        " << basename << "_MESH.h5:/EleConn\n";
	# fs << "                    </DataItem>\n                </Topology>\n";
	# // Write nodal coordinates
	# fs << "\n                <Geometry Type=\"XYZ\">\n                    <DataItem Format=\"HDF\" DataType=\"Float\" Precision=\"8\" Dimensions=\"" << truncateNodeIndex << " 3>\">\n";
	# fs << "\n                        " << basename << "_INC" << incrementID << ".h5:/DefCoords\n";
	# fs << "                    </DataItem>\n                </Geometry>\n                <Time Value=\"" << incrementID << "\" />\n";
	
	# // Cool, lets go and write all the keys
	# vector<string> names = getVariableNames( sdv_bool );

	# for (int i = 0; i < NVARS; i++) {
	#   fs << "                <Attribute Name=\"" << names[i] <<"\" Center=\"Node\">\n                    <DataItem Format=\"HDF\" DataType=\"Float\" Precision=\"8\" Dimensions=\""
	# 	<< truncateNodeIndex << " 1\" >";
	#   fs << "\n                        "<< basename << "_INC" << incrementID << ".h5:/" << names[i] << "\n                    </DataItem>\n                </Attribute>\n";
	# }
	# fs << "\n            </Grid>\n";