import numpy as np
from sklearn.neighbors import KDTree

class mesh:
	"""
	This class holds the node table, element table, 
	and element set table as numpy arrays
	Attributes:
	----------
	node_table : 2D numpy array 
	a collection of [node_id, nodal x,y,z coordinates]
	
	element_table : 2D numpy array
	a collection of [element_id, nodes connected to element_id]
	
	elset_table : 2D numpy array 
	a collection of element sets [grain id, element ids]

	Methods:
	-------
	read_inp(self, inp_name)
	fills the node, element and elset tables

	element_centroid_list(self)
	returns centroid of all elements in element_table

	"""
	def __init__(self, inp_name):
		self.inp_name = inp_name
		self.node_table = np.array([])
		self.element_table = np.array([])
		self.elset_table = np.array([]) 
		self.elset_cs_dict = {}
		self.read_inp(inp_name)
		self.tree = self.generate_KDTree()

	def read_inp(self, inp_name):  
		"""
		Reads Abaqus INP file and generates node table,
		element table and element set table
		Parameters:
		----------
		inp_name : string 
		file name of the INP
		"""
		with open(inp_name, 'r') as inp_file:
			inp_string = inp_file.readlines()
		node_table = []
		element_table = []
		elset_table = []
		count = 0
		cs_dict = {}
		read_node = False
		read_element = False
		read_elset = False
		read_elset_cs = False
		read_nset_cs = False
		for line in inp_string:
			if line[0:5].lower() == '*node':
				print('Reading nodes')
				read_node = True
				continue
			if line[0:13].lower() == '*user element':
				print('Done reading nodes\nTotal no. of nodes:', len(node_table))
				read_node = False
				continue
			if read_node:
				node_table.append([float(i) for i in line.split(',')])
				continue
			if line[0:8].lower() == '*element':
				print('\nReading Elements')
				read_element = True
				continue
			if line[0:5].lower() == '*nset':
				if read_element:
					print('Done reading Elements\nTotal no. of elements:', len(element_table))
				read_element = False          
			if read_element:
				tmp = [int(i) for i in line.split(',')]
				element_table.append([tmp[0], tmp[1], tmp[3], tmp[2], tmp[4]]) #to account for scifen convention
			if line[0:6].lower() == '*elset':
				if read_elset_cs == True:
					print('Done reading element set ', elset_name)
				if line[0:20].lower() != '*elset, elset=grain_':
					read_elset_cs = True
					elset_name = line.split(',')[1][7:]
					cs_dict[elset_name] = []
					print('Reading element set ', elset_name)
					continue
				elif line[0:20].lower() == '*elset, elset=grain_':
					read_elset_cs = False
					# print('Done reading element set for ', elset_name)
			if read_elset_cs == True:
				tempe = [int(i) for i in line.strip().strip('\n').rstrip(',').split(',')]
				# print(tempe)
				cs_dict[elset_name].extend(tempe)
				# print(len([int(i) for i in line.split(',')]))
			if line[0:20].lower() == '*elset, elset=grain_':
				read_elset = True
				count += 1
				if count != 1:
					elset_table.append(elset_line)
				else:
					print('\nReading element sets for grains')
				elset_line = [int(line.split(',')[1][13:])] #GrainID
				continue
			if line[0:9].lower() == '*end part':
				elset_table.append(elset_line)
				print('Done reading element sets\nTotal no. of elsets:', len(elset_table))
				read_elset = False
				break #break here
			if read_elset:
				elset_line.extend([int(i) for i in line.split(',')]) 

			if line[0:36].lower() == '*nset, nset=release_mateside_nodes_0' or \
			line[0:33].lower() == '*nset, nset=crack_front_nodes_all':
				read_nset_cs = False
				print('Done reading node set ', nset_name)
				continue
			if line[0:36].lower() == '*nset, nset=release_mainside_nodes_0':
				read_nset_cs = True
				nset_name = line.split(',')[1][14:]
				cs_dict[nset_name] = []
				print('Reading node set ', nset_name)
				continue
			if line[0:31].lower() == '*nset, nset=crack_front_nodes_0':
				read_nset_cs = True
				nset_name = line.split(',')[1][6:]
				cs_dict[nset_name] = []
				print('Reading node set ', nset_name)
				continue
			if read_nset_cs: 
				tempn = [int(i) for i in line.strip().strip('\n').rstrip(',').split(',')]
				cs_dict[nset_name].extend(tempn)
				continue

		self.node_table = np.array(node_table)
		self.element_table = np.array(element_table)
		# print('elset_table: ')
		# for each in elset_table:
		# 	print(each,len(each),'\n') #added 1 in AI17_S03 inp 100017 and AI50_S01 inp 100117
		self.elset_table = np.array(elset_table) 
		self.elset_cs_dict = cs_dict	            
		return

	def element_centroid_list(self):
		"""
		Calculates the centroid of each element in the element table.
		Node and element table should be available to use this method 
		"""
		if len(self.node_table) == 0 or len(self.element_table) == 0:
			raise Exception('Error: Node or Element table is not yet populated!')
		centroid_list = []
		for element in self.element_table:
			centroid = np.zeros((1,3))
			for i in range(1, 5):
				centroid += self.node_table[element[i]-1,1:4]
			centroid /= 4.0
			centroid_list.append(centroid.ravel())
		assert(len(centroid_list)==len(self.element_table))
		return centroid_list

	def generate_KDTree(self):
	    centroid_list = self.element_centroid_list()
	    tree = KDTree(centroid_list, leaf_size=5, metric='euclidean')
	    return tree


#the  code hereafter is for debugging purpose only
if __name__ == "__main__":
	fe_mesh = mesh('/home/vignesh/projects/CP_simulation/AI1/S00/06-Simulation/AI1_S00.inp')
	print('node table shape', fe_mesh.node_table.shape)
	print(fe_mesh.node_table[0])
	print('element table shape', fe_mesh.element_table.shape)
	print(fe_mesh.element_table[0])
	print('element set table shape', fe_mesh.elset_table.shape)
	print(fe_mesh.elset_table[0])

	set_names = list(fe_mesh.elset_cs_dict.keys())
	print('crack element/node set names: ', set_names)
	print('Length of first crack surface element set: ', len(fe_mesh.elset_cs_dict['LOWER_ELEMENTS_0\n']))
	print(fe_mesh.elset_cs_dict['LOWER_ELEMENTS_0\n'][0])
	print('Length of first crack surface node set: ', len(fe_mesh.elset_cs_dict['MAINSIDE_NODES_0\n']))
	print(fe_mesh.elset_cs_dict['MAINSIDE_NODES_0\n'][0])
	print('Length of crack front node set: ', len(fe_mesh.elset_cs_dict['CRACK_FRONT_NODES_0\n']))
	print(fe_mesh.elset_cs_dict['CRACK_FRONT_NODES_0\n'][0])

	elem_cent = fe_mesh.element_centroid_list()
	print('Length of element centroid list', len(elem_cent))
	print(elem_cent[0])

