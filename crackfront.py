import numpy as np


class crackFront:
	def __init__(self, adv_crk_file=None, ctd_file=None, probepathDB_file=None):
		self.num_cfp = np.nan
		self.origin_loc = []
		self.curr_cf_loc = []
		self.pred_cf_loc = []
		self.gamma = []
		self.kink_angle = []
		self.raw_da = []
		self.smooth_da = []
		self.del_CTD = []
		if adv_crk_file is not None and ctd_file is not None:
			self.read_fill_data(adv_crk_file, ctd_file)
		elif probepathDB_file is not None:
			self.construct_adv_crk_template(probepathDB_file)
		else:
			raise NotImplementedError()

	def read_fill_data(self, adv_crk_file, ctd_file):

		print(f'\nReading Advancing crack file: {adv_crk_file}')
		adv_crk = np.genfromtxt(adv_crk_file, delimiter=',')
		sorted_idx = np.argsort(adv_crk[:,10])
		adv_crk = adv_crk[sorted_idx]
		print('Total no. of crackfront points: ', len(adv_crk))
		self.origin_loc = adv_crk[:,1:4]
		self.curr_cf_loc = adv_crk[:,4:7]
		self.pred_cf_loc = adv_crk[:,7:10]
		self.gamma = adv_crk[:,10].reshape((-1,1))
		self.kink_angle = adv_crk[:,11].reshape((-1,1))
		self.raw_da = np.linalg.norm(np.subtract(self.pred_cf_loc,self.curr_cf_loc), axis=1)
		self.num_cfp = int(len(self.origin_loc))
		

		print(f'\nReading CTD file: {ctd_file}')
		ctd_data = np.genfromtxt(ctd_file, delimiter=',', skip_header=0)
		idx = np.argsort(ctd_data[:,-1])
		ctd_data = ctd_data[idx]

		self.smooth_da = ctd_data[:,3]
		self.del_CTD = ctd_data[:,5] - ctd_data[:,6]
		# assert((self.da == ctd_data[:,4]).all())

		return

	def recalculate_cfp(self, gamma, recalc_what='xyz'):
		"""
		This method calculates (by linear interpolation) the (x,y,z) coordinates of an arbitrary 
		point on the crack front given the angular location of that point along the crack front
		This method does not mutate the crack front object. 
		gamma is the angular location of the crack front point along the crack front (in radians) 
		recalc_what could be 'xyz' or 'cf'
		'xyz' only interpolates and returns the current crack front location
		'cf' interpolates all columns in the crackfront file 
		"""
		recalc_cf = []
		# for gamma in np.arange(-np.pi/2, np.pi/2 + sample_freq*(np.pi/180)/2, sample_freq*(np.pi/180)):
		try:
			after_id = np.where(self.gamma >= gamma)[0][0]
			before_id = after_id-1
		except:
			after_id = np.array([])
		if after_id.size == 0:
			after_id = len(self.gamma)-1
			before_id = len(self.gamma)-1
		if before_id < 0:
			before_id = 0
		if before_id == after_id:
			# print('imhere')
			if recalc_what == 'xyz':
				q = self.curr_cf_loc[before_id]
			elif recalc_what == 'cf':
				q = np.concatenate((self.origin_loc[before_id],self.curr_cf_loc[before_id]\
					,self.pred_cf_loc[before_id],self.gamma[before_id],self.kink_angle[before_id]\
					,[self.raw_da[before_id]], [self.smooth_da[before_id]], [self.del_CTD[before_id]]))
			else:
				raise NotImplementedError('recalc_what could only be "xyz", or "cf"')
			return q
		else:
			q = []
			if recalc_what == 'xyz':
				for before_q, after_q in zip(self.curr_cf_loc[before_id], self.curr_cf_loc[after_id]):
					q.append(before_q.item() + (gamma - self.gamma[before_id].item()) * (after_q.item() - before_q) / \
					(self.gamma[after_id].item() - self.gamma[before_id].item()))

			elif recalc_what == 'cf':
				for before_q, after_q in zip(np.concatenate((self.origin_loc[before_id],\
					self.curr_cf_loc[before_id]	,self.pred_cf_loc[before_id],self.gamma[before_id],\
					self.kink_angle[before_id], [self.raw_da[before_id]], [self.smooth_da[before_id]], [self.del_CTD[before_id]])),\
				np.concatenate((self.origin_loc[after_id],self.curr_cf_loc[after_id] ,self.pred_cf_loc[after_id],\
					self.gamma[after_id], self.kink_angle[after_id], [self.raw_da[after_id]], [self.smooth_da[after_id]], [self.del_CTD[after_id]]))):

					q.append(before_q.item() + (gamma - self.gamma[before_id].item()) * (after_q.item() - before_q) / \
					(self.gamma[after_id].item() - self.gamma[before_id].item()))
			else:
				raise NotImplementedError('recalc_what could only be "xyz", or "cf"')
			return(np.array(q))



	def recalculate_cf(self, sample_freq):
		"""
		This method redefines already created crackfront object
		NOTE: this method mutates the crackFront object
		"""
		recalc_cf = []
		for gamma in np.arange(-np.pi/2, np.pi/2 + sample_freq*(np.pi/180)/2, sample_freq*(np.pi/180)):
			try:
				after_id = np.where(self.gamma >= gamma)[0][0]
				before_id = after_id-1
			except:
				after_id = np.array([])
			if after_id.size == 0:
				after_id = len(self.gamma)-1
				before_id = len(self.gamma)-1
			if before_id < 0:
				before_id = 0
			if before_id == after_id:
				# print('imhere')
				q = np.concatenate((self.origin_loc[before_id],self.curr_cf_loc[before_id]\
					,self.pred_cf_loc[before_id],self.gamma[before_id],self.kink_angle[before_id]\
					,[self.raw_da[before_id]], [self.smooth_da[before_id]], [self.del_CTD[before_id]]))
			else:
				q = []
				# print('entERED')
				for before_q, after_q in zip(np.concatenate((self.origin_loc[before_id],\
					self.curr_cf_loc[before_id]	,self.pred_cf_loc[before_id],self.gamma[before_id],\
					self.kink_angle[before_id], [self.raw_da[before_id]], [self.smooth_da[before_id]], [self.del_CTD[before_id]])),\
				np.concatenate((self.origin_loc[after_id],self.curr_cf_loc[after_id] ,self.pred_cf_loc[after_id],\
					self.gamma[after_id], self.kink_angle[after_id], [self.raw_da[after_id]], [self.smooth_da[after_id]], [self.del_CTD[after_id]]))):

					q.append(before_q.item() + (gamma - self.gamma[before_id].item()) * (after_q.item() - before_q) / \
					(self.gamma[after_id].item() - self.gamma[before_id].item()))

			recalc_cf.append(q)
		recalc_cf = np.array(recalc_cf)

		#mutating the object
		self.num_cfp = len(recalc_cf)
		self.origin_loc = recalc_cf[:,0:3]
		self.curr_cf_loc = recalc_cf[:,3:6]
		self.pred_cf_loc = recalc_cf[:,6:9]
		self.gamma = recalc_cf[:,9].reshape(-1,1)
		self.kink_angle = recalc_cf[:,10].reshape(-1,1)
		self.raw_da = recalc_cf[:,11]
		self.smooth_da = recalc_cf[:,12]
		self.del_CTD = recalc_cf[:,13]

		return

	def construct_adv_crk_template(self, probepathDB_file):
		f = open(probepathDB_file, 'r')
		ppdb_content = f.readlines()
		origin = []
		current = []
		gam = []
		for line in ppdb_content:
			temp = line.split(',')
			if temp[0] == 'DEF':
				origin.append([float(temp[3]), float(temp[4]), float(temp[5])])
				current.append([float(temp[6]), float(temp[7]), float(temp[8])])
				gam.append(float(temp[9]))
		origin = np.array(origin)
		current = np.array(current)
		gam = np.array(gam)
		f.close()
		self.num_cfp = len(origin)
		self.origin_loc = origin[:,0:3]
		self.curr_cf_loc = current[:,0:3]
		self.pred_cf_loc = np.tile(np.nan, (len(origin),3))
		self.gamma = gam.reshape(-1,1)
		self.kink_angle = np.tile(np.nan, (len(origin),1))
		return

	def write_adv_crack_file(self, write_path, write_sorted=False):
		f = open(write_path, 'w')
		for i in range(self.num_cfp):
			f.write(str(0)+','+str(self.origin_loc[i,0])+','+str(self.origin_loc[i,1])+','+str(self.origin_loc[i,2])\
				+','+str(self.curr_cf_loc[i,0])+','+str(self.curr_cf_loc[i,1])+','+str(self.curr_cf_loc[i,2])\
				+','+str(self.pred_cf_loc[i,0])+','+str(self.pred_cf_loc[i,1])+','+str(self.pred_cf_loc[i,2])\
				+','+str(self.gamma[i,0])+','+str(self.kink_angle[i,0])+'\n')
		f.close()
		print('Successfully wrote AdvancingCrack_generated file!')

	def fill_adv_crk_template(self, gammas, kas, das):
		raw_da = []
		for j, g in enumerate(self.gamma): 
			ka, da = self.reinterpolate_ka_da(g.item(), gammas.ravel(), kas.ravel(), das.ravel())
			self.kink_angle[j,0] = ka
			raw_da.append(da)
			self.pred_cf_loc[j,0:3] = self.calc_pred_cf_loc(self.curr_cf_loc[j,0:3], g.item(), ka, da)
		self.raw_da = np.array(raw_da).reshape(-1,1)
		return

	def calc_pred_cf_loc(self, curr_cf_loc, g, ka, da):
		R1 = np.matrix([ [np.cos(g), -np.sin(g), 0] , [np.sin(g), np.cos(g), 0] , [0,0,1] ])
		R2 = np.matrix([ [1,0,0] , [0 , np.cos(ka), -np.sin(ka)] , [0 , np.sin(ka), np.cos(ka)] ])
		ini_vec = np.array([0,1,0]).reshape(3,1) # y axis
		final_vec = np.matmul(np.matmul( R1 , R2 ) , ini_vec)
		final_loc = np.array(curr_cf_loc) + da * final_vec.A1
		return final_loc

	def reinterpolate_ka_da(self, g, gammas, kas, das):
		sort_idx = np.argsort(gammas)
		gammas = gammas[sort_idx]
		kas = kas[sort_idx]
		das = das[sort_idx]

		try:
			after_id = np.where(gammas > g)[0][0]
			before_id = after_id-1
		except:
			after_id = np.array([])
		if after_id.size == 0:
			after_id = len(gammas)-1
			before_id = len(gammas)-1
		if before_id < 0:
			before_id = 0
		if before_id == after_id: # no need to interpolate
			ka = kas[before_id]
			da = das[before_id]
		else: # need to interpolate ka and da
			ka = kas[before_id] + (g - gammas[before_id]) * (kas[after_id]-kas[before_id]) / (gammas[after_id]-gammas[before_id])
			da = das[before_id] + (g - gammas[before_id]) * (das[after_id]-das[before_id]) / (gammas[after_id]-gammas[before_id])

		return ka, da

#for debugging
if __name__ == '__main__':
	cf = crackFront('/media/vignesh/easystore/CP_simulation/AI2/S03/06-Simulation/AdvancingCrack_exp',\
	 '/media/vignesh/easystore/CP_simulation/AI2/S03/06-Simulation/CTD_mags_smooth_1')
	print('Before recalculating crackfront:\n')
	print('num_cfp: ', cf.num_cfp, 'type: ', type(cf.num_cfp))
	print('origin_loc: ', cf.origin_loc[1], 'type: ', type(cf.origin_loc[1]))
	print('curr_cf_loc: ', cf.curr_cf_loc[1], 'type: ', type(cf.curr_cf_loc[1]))
	print('pred_cf_loc: ', cf.pred_cf_loc[1], 'type: ', type(cf.pred_cf_loc[1]))
	print('gamma: ', cf.gamma[1], 'type: ', type(cf.gamma[1]))
	print('kink_angle: ', cf.kink_angle[1], 'type: ', type(cf.kink_angle[1]))
	print('raw_da: ', cf.raw_da[1], 'type: ', type(cf.raw_da[1]))
	print('smooth_da: ', cf.smooth_da[1], 'type: ', type(cf.smooth_da[1]))
	cf.recalculate_cf(5)
	print('After recalculating crackfront:\n')
	print('num_cfp: ', cf.num_cfp, 'type: ', type(cf.num_cfp))
	print('origin_loc: ', cf.origin_loc[1], 'type: ', type(cf.origin_loc[1]))
	print('curr_cf_loc: ', cf.curr_cf_loc[1], 'type: ', type(cf.curr_cf_loc[1]))
	print('pred_cf_loc: ', cf.pred_cf_loc[1], 'type: ', type(cf.pred_cf_loc[1]))
	print('gamma: ', cf.gamma[1], 'type: ', type(cf.gamma[1]))
	print('kink_angle: ', cf.kink_angle[1], 'type: ', type(cf.kink_angle[1]))
	print('raw_da: ', cf.raw_da[1], 'type: ', type(cf.raw_da[1]))
	print('smooth_da: ', cf.smooth_da[1], 'type: ', type(cf.smooth_da[1]))

	#testing recalculate_cfp()
	interp_cur_loc = cf.recalculate_cfp(gamma=5, recalc_what='xyz')
	interp_cfp = cf.recalculate_cfp(gamma=5*(np.pi/180), recalc_what='cf')
	print('interpolated current crack front location: ', interp_cur_loc, ' Its type: ', type(interp_cur_loc))
	print('interpolated all columns for the current crack front point: ', interp_cfp, ' Its type: ', type(interp_cfp))
	# cf.recalculate_cfp(gamma=5, recalc_what='pred_cf_loc')
	print('TESTING ADVANCING CRACK FILE GENERATION FROM PROBEPATHDB\n')
	cf = crackFront(probepathDB_file='/home/vignesh/projects/CP_simulation/no_odb/AI1/S02/05-WC/ProbePathDB')
	cf.write_adv_crack_file('/home/vignesh/projects/CP_simulation/no_odb')

