import numpy as np
from PIL import Image
import cv2
import random
from random import randrange



class Cube:
	def __init__(self):

		self.done = 1
		self.reward = 0

		self.action_space = 19
		self.state_space = 324


		# Initial Orientation
		self.initOrientation = [
			[1,0,0,0,0,0],
			[1,0,0,0,0,0],
			[1,0,0,0,0,0],
			[1,0,0,0,0,0],
			[1,0,0,0,0,0],
			[1,0,0,0,0,0],
			[1,0,0,0,0,0],
			[1,0,0,0,0,0],
			[1,0,0,0,0,0],
			[0,1,0,0,0,0],
			[0,1,0,0,0,0],
			[0,1,0,0,0,0],
			[0,0,1,0,0,0],
			[0,0,1,0,0,0],
			[0,0,1,0,0,0],
			[0,0,0,1,0,0],
			[0,0,0,1,0,0],
			[0,0,0,1,0,0],
			[0,1,0,0,0,0],
			[0,1,0,0,0,0],
			[0,1,0,0,0,0],
			[0,0,1,0,0,0],
			[0,0,1,0,0,0],
			[0,0,1,0,0,0],
			[0,0,0,1,0,0],
			[0,0,0,1,0,0],
			[0,0,0,1,0,0],
			[0,1,0,0,0,0],
			[0,1,0,0,0,0],
			[0,1,0,0,0,0],
			[0,0,1,0,0,0],
			[0,0,1,0,0,0],
			[0,0,1,0,0,0],
			[0,0,0,1,0,0],
			[0,0,0,1,0,0],
			[0,0,0,1,0,0],
			[0,0,0,0,1,0],
			[0,0,0,0,1,0],
			[0,0,0,0,1,0],
			[0,0,0,0,1,0],
			[0,0,0,0,1,0],
			[0,0,0,0,1,0],
			[0,0,0,0,1,0],
			[0,0,0,0,1,0],
			[0,0,0,0,1,0],
			[0,0,0,0,0,1],
			[0,0,0,0,0,1],
			[0,0,0,0,0,1],
			[0,0,0,0,0,1],
			[0,0,0,0,0,1],
			[0,0,0,0,0,1],
			[0,0,0,0,0,1],
			[0,0,0,0,0,1],
			[0,0,0,0,0,1]]

		self.orientation = self.initOrientation

		self.orientation_flat = [i for subOrientation in self.orientation for i in subOrientation]

		# self.colors = {
		# 			'O': (0,165,255),
		# 			'B': (255, 0, 0),
		# 			'Y': (0, 255, 255),
		# 			'G': (0, 255, 0),
		# 			'R': (0, 0, 255),
		# 			'W': (255, 255, 255)}

		self.colors = [
					(0,165,255), # O
					(255, 0, 0), # B
					(0, 255, 255), # Y
					(0, 255, 0), # G
					(0, 0, 255), # R
					(255, 255, 255) # W
				]

		self.color_node_dic = {
					'O': 0,
					'B': 1,
					'Y': 2,
					'G': 3,
					'R': 4,
					'W': 5}




	def is_correct_orientation(self):
		"""
		Check to see if every face has the same color grouping

		:return: <int> Whether the cube is solved
		"""

		# Define the face groups
		f1 = [0, 1, 2, 3, 4, 5, 6, 7, 8] # Back
		f2 = [9, 10, 11, 18, 19, 20, 27, 28, 29] # Left
		f3 = [12, 13, 14, 21, 22, 23, 30, 31, 32] # Top
		f4 = [15, 16, 17, 24, 25, 26, 33, 34, 35] # Right
		f5 = [36, 37, 38, 39, 40, 41, 42, 43, 44] # Front
		f6 = [45, 46, 47, 48, 49, 50, 51, 52, 53] # Bottom

		# Set the inital response to True so we can prove false in the iterations
		solved = True

		# Iterate through each list and check whether all are the same color
		for face in [f1, f2, f3, f4, f5, f6]:
			# Get the position of the '1' for each square in the face
			pos = [self.orientation[s].index(1) for s in face]
			# Check whether all of the positions (colors) are the same for the face
			solved = pos.count(pos[0]) == len(pos)
			if not solved:
				return solved

		self.done = solved
		return solved




		return self.correct_orientation == self.orientation


	def flat_orientation_to_orientation(self, fltLst):
		"""
		Take a flattened orientation list and
		update the cube orientation

		:param fltLst: <list> The flatten list of orientation
		"""

		self.orientation = [fltLst[i:i+6] for i in range(0, len(fltLst), 6)]


	def update_flat_orientation(self):
		"""
		Update the flat orientation list
		"""

		# Update the flattened orientation list
		self.orientation_flat = [i for subOrientation in self.orientation for i in subOrientation]


	def render(self, timeDisplay = 4):
		img = self.get_image()
		img = img.resize((300, 300), resample = Image.NEAREST)  # resizing so we can see our agent in all its glory.
		# print(img)
		cv2.imshow("image", np.array(img))  # show it!
		cv2.waitKey(timeDisplay * 1000)


	def get_image(self):
		
		env = np.zeros((12, 9, 3), dtype=np.uint8)  # starts an rbg of our size
		env[0][3] = self.colors[self.orientation[0].index(1)]
		env[0][4] = self.colors[self.orientation[1].index(1)]
		env[0][5] = self.colors[self.orientation[2].index(1)]
		env[1][3] = self.colors[self.orientation[3].index(1)]
		env[1][4] = self.colors[self.orientation[4].index(1)]
		env[1][5] = self.colors[self.orientation[5].index(1)]
		env[2][3] = self.colors[self.orientation[6].index(1)]
		env[2][4] = self.colors[self.orientation[7].index(1)]
		env[2][5] = self.colors[self.orientation[8].index(1)]
		env[3][0] = self.colors[self.orientation[9].index(1)]
		env[3][1] = self.colors[self.orientation[10].index(1)]
		env[3][2] = self.colors[self.orientation[11].index(1)]
		env[3][3] = self.colors[self.orientation[12].index(1)]
		env[3][4] = self.colors[self.orientation[13].index(1)]
		env[3][5] = self.colors[self.orientation[14].index(1)]
		env[3][6] = self.colors[self.orientation[15].index(1)]
		env[3][7] = self.colors[self.orientation[16].index(1)]
		env[3][8] = self.colors[self.orientation[17].index(1)]
		env[4][0] = self.colors[self.orientation[18].index(1)]
		env[4][1] = self.colors[self.orientation[19].index(1)]
		env[4][2] = self.colors[self.orientation[20].index(1)]
		env[4][3] = self.colors[self.orientation[21].index(1)]
		env[4][4] = self.colors[self.orientation[22].index(1)]
		env[4][5] = self.colors[self.orientation[23].index(1)]
		env[4][6] = self.colors[self.orientation[24].index(1)]
		env[4][7] = self.colors[self.orientation[25].index(1)]
		env[4][8] = self.colors[self.orientation[26].index(1)]
		env[5][0] = self.colors[self.orientation[27].index(1)]
		env[5][1] = self.colors[self.orientation[28].index(1)]
		env[5][2] = self.colors[self.orientation[29].index(1)]
		env[5][3] = self.colors[self.orientation[30].index(1)]
		env[5][4] = self.colors[self.orientation[31].index(1)]
		env[5][5] = self.colors[self.orientation[32].index(1)]
		env[5][6] = self.colors[self.orientation[33].index(1)]
		env[5][7] = self.colors[self.orientation[34].index(1)]
		env[5][8] = self.colors[self.orientation[35].index(1)]
		env[6][3] = self.colors[self.orientation[36].index(1)]
		env[6][4] = self.colors[self.orientation[37].index(1)]
		env[6][5] = self.colors[self.orientation[38].index(1)]
		env[7][3] = self.colors[self.orientation[39].index(1)]
		env[7][4] = self.colors[self.orientation[40].index(1)]
		env[7][5] = self.colors[self.orientation[41].index(1)]
		env[8][3] = self.colors[self.orientation[42].index(1)]
		env[8][4] = self.colors[self.orientation[43].index(1)]
		env[8][5] = self.colors[self.orientation[44].index(1)]
		env[9][3] = self.colors[self.orientation[45].index(1)]
		env[9][4] = self.colors[self.orientation[46].index(1)]
		env[9][5] = self.colors[self.orientation[47].index(1)]
		env[10][3] = self.colors[self.orientation[48].index(1)]
		env[10][4] = self.colors[self.orientation[49].index(1)]
		env[10][5] = self.colors[self.orientation[50].index(1)]
		env[11][3] = self.colors[self.orientation[51].index(1)]
		env[11][4] = self.colors[self.orientation[52].index(1)]
		env[11][5] = self.colors[self.orientation[53].index(1)]


		img = Image.fromarray(env, 'RGB')
		return img


	def shift_orientation(self, lst):
		"""
		Shift the orientation of the elements found in positions
		given by a list.
		
		lst <list>: The positions that needs shifting

		:return: None
		"""
		

		# Store first element value
		temp = self.orientation[lst[0]]

		# Get elements from the proceding position
		# but not for the last element
		for i in range(len(lst) - 1):
			self.orientation[lst[i]] = self.orientation[lst[i + 1]]

		# Last element filled with the temp value
		self.orientation[lst[len(lst) - 1]] = temp


	# Define movements

	def move_R(self):
		"""
		| | ^
		| | |
		| | |
		"""
		self.shift_orientation([2, 14, 38, 47])
		self.shift_orientation([5, 23, 41, 50])
		self.shift_orientation([8, 32, 44, 53])

		self.shift_orientation([17, 15, 33, 35])
		self.shift_orientation([16, 24, 34, 26])

		self.is_correct_orientation()

		self.update_flat_orientation()

	def move_Rp(self):
		"""
		| | |
		| | |
		| | v
		"""
		self.shift_orientation([47, 38, 14, 2])
		self.shift_orientation([50, 41, 23, 5])
		self.shift_orientation([53, 44, 32, 8])

		self.shift_orientation([35, 33, 15, 17])
		self.shift_orientation([26, 34, 24, 16])
		
		self.is_correct_orientation()

		self.update_flat_orientation()

	def move_C(self):
		"""
		| ^ |
		| | |
		| | |
		"""
		
		self.shift_orientation([1, 13, 37, 46])
		self.shift_orientation([4, 22, 40, 49])
		self.shift_orientation([7, 31, 43, 52])
		
		self.is_correct_orientation()

		self.update_flat_orientation()

	def move_Cp(self):
		"""
		| | |
		| | |
		| v |
		"""
		
		self.shift_orientation([46, 37, 13, 1])
		self.shift_orientation([49, 40, 22, 4])
		self.shift_orientation([52, 43, 31, 7])
		
		self.is_correct_orientation()

		self.update_flat_orientation()

	def move_L(self):
		"""
		^ | |
		| | |
		| | |
		"""
		self.shift_orientation([0, 12, 36, 45])
		self.shift_orientation([3, 21, 39, 48])
		self.shift_orientation([6, 30, 42, 51])

		self.shift_orientation([9, 11, 29, 27])
		self.shift_orientation([10, 20, 28, 18])
		
		self.is_correct_orientation()

		self.update_flat_orientation()

	def move_Lp(self):
		"""
		| | |
		| | |
		v | |
		"""
		self.shift_orientation([45, 36, 12, 0])
		self.shift_orientation([48, 39, 21, 3])
		self.shift_orientation([51, 42, 30, 6])

		self.shift_orientation([27, 29, 11, 9])
		self.shift_orientation([18, 28, 20, 10])
		
		self.is_correct_orientation()

		self.update_flat_orientation()


	def move_U(self):
		"""
		< - -
		- - -
		- - -
		"""
		self.shift_orientation([12, 15, 53, 9])
		self.shift_orientation([13, 16, 52, 10])
		self.shift_orientation([14, 17, 51, 11])

		self.shift_orientation([6, 8, 2, 0])
		self.shift_orientation([7, 5, 1, 3])
		
		self.is_correct_orientation()

		self.update_flat_orientation()

	def move_Up(self):
		"""
		- - >
		- - -
		- - -
		"""
		self.shift_orientation([9, 53, 15, 12])
		self.shift_orientation([10, 52, 16, 13])
		self.shift_orientation([11, 51, 17, 14])

		self.shift_orientation([0, 2, 8, 6])
		self.shift_orientation([3, 1, 5, 7])
		
		self.is_correct_orientation()

		self.update_flat_orientation()

	def move_M(self):
		"""
		- - -
		< - -
		- - -
		"""
		
		self.shift_orientation([21, 24, 50, 18])
		self.shift_orientation([22, 25, 49, 19])
		self.shift_orientation([23, 26, 48, 20])
		
		self.is_correct_orientation()

		self.update_flat_orientation()

	def move_Mp(self):
		"""
		- - -
		- - >
		- - -
		"""
		
		self.shift_orientation([18, 50, 24, 21])
		self.shift_orientation([19, 49, 25, 22])
		self.shift_orientation([20, 48, 26, 23])
		
		self.is_correct_orientation()

		self.update_flat_orientation()

	def move_B(self):
		"""
		- - -
		- - -
		- - >
		"""
		self.shift_orientation([32, 29, 45, 35])
		self.shift_orientation([31, 28, 46, 34])
		self.shift_orientation([30, 27, 47, 33])

		self.shift_orientation([43, 41, 37, 39])
		self.shift_orientation([44, 38, 36, 42])
		
		self.is_correct_orientation()

		self.update_flat_orientation()

	def move_Bp(self):
		"""
		- - -
		- - -
		< - -
		"""
		self.shift_orientation([35, 45, 29, 32])
		self.shift_orientation([34, 46, 28, 31])
		self.shift_orientation([33, 47, 27, 30])

		self.shift_orientation([39, 37, 41, 43])
		self.shift_orientation([42, 36, 38, 44])
		
		self.is_correct_orientation()

		self.update_flat_orientation()



	def move_F(self):
		"""
		- - -
		^   v
		- - -
		"""
		
		self.shift_orientation([8, 11, 36, 33])
		self.shift_orientation([7, 20, 37, 24])
		self.shift_orientation([6, 29, 38, 15])

		self.shift_orientation([32, 14, 12, 30])
		self.shift_orientation([23, 13, 21, 31])
		
		self.is_correct_orientation()

		self.update_flat_orientation()

	def move_Fp(self):
		"""
		- - -
		v   ^
		- - -
		"""
		
		self.shift_orientation([33, 36, 11, 8])
		self.shift_orientation([24, 37, 20, 7])
		self.shift_orientation([15, 38, 29, 6])

		self.shift_orientation([30, 12, 14, 32])
		self.shift_orientation([31, 21, 13, 23])
		
		self.is_correct_orientation()

		self.update_flat_orientation()

	def move_I(self):
		"""
		- 	- 	-

		v Inner  ^

		- 	- 	-
		"""
		
		self.shift_orientation([21, 18, 50, 24])
		self.shift_orientation([22, 19, 49, 25])
		self.shift_orientation([23, 20, 48, 26])
		
		self.is_correct_orientation()

		self.update_flat_orientation()

	def move_Ip(self):
		"""
		- 	- 	-

		^ Inner v

		- 	- 	-
		"""
		
		self.shift_orientation([21, 24, 50, 18])
		self.shift_orientation([22, 25, 49, 19])
		self.shift_orientation([23, 26, 48, 20])
		
		self.is_correct_orientation()

		self.update_flat_orientation()

	def move_O(self):
		"""
		- 		- 	-

		v Backside  ^

		- 		- 	-
		"""
		
		self.shift_orientation([0, 17, 44, 27])
		self.shift_orientation([1, 26, 43, 18])
		self.shift_orientation([2, 35, 42, 9])

		self.shift_orientation([53, 47, 45, 51])
		self.shift_orientation([50, 46, 48, 52])
		
		self.is_correct_orientation()

		self.update_flat_orientation()

	def move_Op(self):
		"""
		- 		- 	-

		^ Backside  v

		- 		- 	-
		"""
		
		self.shift_orientation([27, 44, 17, 0])
		self.shift_orientation([18, 43, 26, 1])
		self.shift_orientation([9, 42, 35, 2])

		self.shift_orientation([51, 45, 47, 53])
		self.shift_orientation([52, 48, 46, 50])
		
		self.is_correct_orientation()

		self.update_flat_orientation()


	def reset(self):
		"""
		Reset the positions of the cube to the original state
		:return: <list> flattened cube orientation
		"""

		self.orientation = self.initOrientation

		self.update_flat_orientation()

		return self.orientation_flat


	def shuffle(self, n):
		"""
		Randomly shuffle the cube by turning
		it n number of times

		:param n: <int> Integer for number of time to make a turn
		:return: <list> cube orientation
		"""

		# Reset the cube to the correct orientation first
		self.reset()

		for i in range(n):
			# Get the random move from 0-17
			move = randrange(18)

			if move == 0:
				self.move_R()
			elif move == 1:
				self.move_Rp()
			elif move == 2:
				self.move_C()
			elif move == 3:
				self.move_Cp()
			elif move == 4:
				self.move_L()
			elif move == 5:
				self.move_Lp()
			elif move == 6:
				self.move_U()
			elif move == 7:
				self.move_Up()
			elif move == 8:
				self.move_M()
			elif move == 9:
				self.move_Mp()
			elif move == 10:
				self.move_B()
			elif move == 11:
				self.move_Bp()
			elif move == 12:
				self.move_F()
			elif move == 13:
				self.move_Fp()
			elif move == 14:
				self.move_I()
			elif move == 15:
				self.move_Ip()
			elif move == 16:
				self.move_O()
			elif move == 17:
				self.move_Op()

		return self.orientation_flat


	def step(self, action):
		"""
		Make a turn on the cube given an action number
		"""

		self.reward = 0
		self.done = False

		if action == 0:
			self.move_R()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 1:
			self.move_Rp()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 2:
			self.move_C()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 3:
			self.move_Cp()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 4:
			self.move_L()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 5:
			self.move_Lp()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 6:
			self.move_U()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 7:
			self.move_Up()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 8:
			self.move_M()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 9:
			self.move_Mp()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 10:
			self.move_B()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 11:
			self.move_Bp()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 12:
			self.move_F()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 13:
			self.move_Fp()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 14:
			self.move_I()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 15:
			self.move_Ip()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 16:
			self.move_O()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 17:
			self.move_Op()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1

		return self.reward, self.orientation_flat, self.done



if __name__ == '__main__':
	c = Cube()
	# print(c.orientation_flat)
	# print(len(c.orientation_flat))
	c.render(2)
	print(c.step(14))
	c.render(2)
	print(c.step(15))
	c.render(2)





























	# # for s in range(1000000):
	# # 	c.move_R()
	# # 	c.move_M()
	# # 	i += 1
	# # 	if c.is_correct_orientation():
	# # 		print(i)
	# # 		break

	
	# c.move_R()
	# c.move_U()
	# # c.move_C()

	# mapping_dic = dict(list(enumerate(c.orientationN)))
	
	# mapping_dic_clean = {k:v for k,v in mapping_dic.items() if k != v}


	# groups = []

	# groupList = []

	# def create_group(value, group = None):
	# 	if group is None:
	# 		group = []
	# 	if value not in [item for sublist in groups for item in sublist] and value not in group:
	# 		group.append(value)
	# 		create_group(mapping_dic[value], group)
	# 	else:
	# 		if group: # Don't add any empty lists
	# 			return groups.append(group)

	# for i in mapping_dic_clean: # mapping_dic_clean for no singularities, mapping_dic for singularities included.
	# 	create_group(i)


	# print(groups)
	# print(mapping_dic_clean)
	# c.render(10)