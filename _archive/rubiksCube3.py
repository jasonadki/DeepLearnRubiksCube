import numpy as np
from PIL import Image
import cv2
import random



class Cube:
	def __init__(self):
		self.orientation = [
						'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y',
						'B', 'B', 'B', 'R', 'R', 'R', 'G', 'G', 'G',
						'B', 'B', 'B', 'R', 'R', 'R', 'G', 'G', 'G',
						'B', 'B', 'B', 'R', 'R', 'R', 'G', 'G', 'G',
						'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W',
						'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'
					   ]

		self.correct_orientation = [
						'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y',
						'B', 'B', 'B', 'R', 'R', 'R', 'G', 'G', 'G',
						'B', 'B', 'B', 'R', 'R', 'R', 'G', 'G', 'G',
						'B', 'B', 'B', 'R', 'R', 'R', 'G', 'G', 'G',
						'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W',
						'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'
					   ]

		self.orientationN = [i for i in range(54)]
		self.correct_orientationN = [i for i in range(54)]

		self.colors = {
					'O': (0,165,255),
					'B': (255, 0, 0),
					'Y': (0, 255, 255),
					'G': (0, 255, 0),
					'R': (0, 0, 255),
					'W': (255, 255, 255)
				}

		self.color_node_dic = {
					'O': 0,
					'B': 1,
					'Y': 2,
					'G': 3,
					'R': 4,
					'W': 5		
		}

		self.node_orientation = [[0 for i in range(6)] for j in range(54)]
		self.update_node_orientation()


	def update_node_orientation(self):

		self.node_orientation = [[0 for i in range(6)] for j in range(54)]

		for i in enumerate(self.orientation):
			color_position = self.color_node_dic[i[1]]
			self.node_orientation[i[0]][color_position] = 1

	def is_correct_orientation(self):
		return self.correct_orientation == self.orientation

	def is_correct_orientationN(self):
		return self.correct_orientationN == self.orientationN

	def render(self, timeDisplay = 4):
		img = self.get_image()
		img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
		# print(img)
		cv2.imshow("image", np.array(img))  # show it!
		cv2.waitKey(timeDisplay * 1000)

	def get_image(self):
		
		env = np.zeros((12, 9, 3), dtype=np.uint8)  # starts an rbg of our size
		env[0][3] = self.colors[self.orientation[0]]
		env[0][4] = self.colors[self.orientation[1]]
		env[0][5] = self.colors[self.orientation[2]]
		env[1][3] = self.colors[self.orientation[3]]
		env[1][4] = self.colors[self.orientation[4]]
		env[1][5] = self.colors[self.orientation[5]]
		env[2][3] = self.colors[self.orientation[6]]
		env[2][4] = self.colors[self.orientation[7]]
		env[2][5] = self.colors[self.orientation[8]]
		env[3][0] = self.colors[self.orientation[9]]
		env[3][1] = self.colors[self.orientation[10]]
		env[3][2] = self.colors[self.orientation[11]]
		env[3][3] = self.colors[self.orientation[12]]
		env[3][4] = self.colors[self.orientation[13]]
		env[3][5] = self.colors[self.orientation[14]]
		env[3][6] = self.colors[self.orientation[15]]
		env[3][7] = self.colors[self.orientation[16]]
		env[3][8] = self.colors[self.orientation[17]]
		env[4][0] = self.colors[self.orientation[18]]
		env[4][1] = self.colors[self.orientation[19]]
		env[4][2] = self.colors[self.orientation[20]]
		env[4][3] = self.colors[self.orientation[21]]
		env[4][4] = self.colors[self.orientation[22]]
		env[4][5] = self.colors[self.orientation[23]]
		env[4][6] = self.colors[self.orientation[24]]
		env[4][7] = self.colors[self.orientation[25]]
		env[4][8] = self.colors[self.orientation[26]]
		env[5][0] = self.colors[self.orientation[27]]
		env[5][1] = self.colors[self.orientation[28]]
		env[5][2] = self.colors[self.orientation[29]]
		env[5][3] = self.colors[self.orientation[30]]
		env[5][4] = self.colors[self.orientation[31]]
		env[5][5] = self.colors[self.orientation[32]]
		env[5][6] = self.colors[self.orientation[33]]
		env[5][7] = self.colors[self.orientation[34]]
		env[5][8] = self.colors[self.orientation[35]]
		env[6][3] = self.colors[self.orientation[36]]
		env[6][4] = self.colors[self.orientation[37]]
		env[6][5] = self.colors[self.orientation[38]]
		env[7][3] = self.colors[self.orientation[39]]
		env[7][4] = self.colors[self.orientation[40]]
		env[7][5] = self.colors[self.orientation[41]]
		env[8][3] = self.colors[self.orientation[42]]
		env[8][4] = self.colors[self.orientation[43]]
		env[8][5] = self.colors[self.orientation[44]]
		env[9][3] = self.colors[self.orientation[45]]
		env[9][4] = self.colors[self.orientation[46]]
		env[9][5] = self.colors[self.orientation[47]]
		env[10][3] = self.colors[self.orientation[48]]
		env[10][4] = self.colors[self.orientation[49]]
		env[10][5] = self.colors[self.orientation[50]]
		env[11][3] = self.colors[self.orientation[51]]
		env[11][4] = self.colors[self.orientation[52]]
		env[11][5] = self.colors[self.orientation[53]]


		img = Image.fromarray(env, 'RGB')
		return img

	def shift_orientation(self, lst):
		"""
		Shift the orientation of the elements found in positions
		given by a list.
		
		lst <list>: The positions that needs shifting

		:return: None
		"""
		
		###################
		# Color Orientation
		###################

		# Store first element value
		temp = self.orientation[lst[0]]

		# Get elements from the proceding position
		# but not for the last element
		for i in range(len(lst) - 1):
			self.orientation[lst[i]] = self.orientation[lst[i + 1]]

		# Last element filled with the temp value
		self.orientation[lst[len(lst) - 1]] = temp



		####################
		# Number Orientation
		####################

		# Store first element value
		tempN = self.orientationN[lst[0]]

		# Get elements from the proceding position
		# but not for the last element
		for i in range(len(lst) - 1):
			# print(f'Putting {self.orientationN[lst[i + 1]]} into {self.orientationN[lst[i]]}')
			self.orientationN[lst[i]] = self.orientationN[lst[i + 1]]

		# Last element filled with the temp value
		# print(f'Putting {tempN} into {self.orientationN[lst[len(lst) - 1]]}')
		self.orientationN[lst[len(lst) - 1]] = tempN

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

		self.update_node_orientation()

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

		self.update_node_orientation()

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

		self.update_node_orientation()

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

		self.update_node_orientation()

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

		self.update_node_orientation()

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

		self.update_node_orientation()

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

		self.update_node_orientation()

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

		self.update_node_orientation()

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

		self.update_node_orientation()

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

		self.update_node_orientation()

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

		self.update_node_orientation()

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

		self.update_node_orientation()

	def move_M(self):
		"""
		- - -
		< - -
		- - -
		"""
		
		self.shift_orientation([21, 24, 50, 18])
		self.shift_orientation([22, 25, 49, 19])
		self.shift_orientation([23, 26, 48, 20])

		self.update_node_orientation()

	def move_Mp(self):
		"""
		- - -
		- - >
		- - -
		"""
		
		self.shift_orientation([18, 50, 24, 21])
		self.shift_orientation([19, 49, 25, 22])
		self.shift_orientation([20, 48, 26, 23])

		self.update_node_orientation()

	def move_C(self):
		"""
		| ^ |
		| | |
		| | |
		"""
		
		self.shift_orientation([1, 13, 37, 46])
		self.shift_orientation([4, 22, 40, 49])
		self.shift_orientation([7, 31, 43, 52])

		self.update_node_orientation()

	def move_Cp(self):
		"""
		| | |
		| | |
		| v |
		"""
		
		self.shift_orientation([46, 37, 13, 1])
		self.shift_orientation([49, 40, 22, 4])
		self.shift_orientation([52, 43, 31, 7])

		self.update_node_orientation()

	def move_random(self, num):
		if num == 0:
			self.move_R()
		elif num == 1:
			self.move_Rp()
		elif num == 2:
			self.move_L()
		elif num == 3:
			self.move_Lp()
		elif num == 4:
			self.move_U()
		elif num == 5:
			self.move_Up()
		elif num == 6:
			self.move_B()
		elif num == 7:
			self.move_Bp()
		elif num == 8:
			self.move_F()
		elif num == 9:
			self.move_Fp()
		elif num == 10:
			self.move_O()
		elif num == 11:
			self.move_Op()
		elif num == 12:
			self.move_M()
		elif num == 13:
			self.move_Mp()
		elif num == 14:
			self.move_C()
		elif num == 15:
			self.move_Cp()


if __name__ == '__main__':
	c = Cube()
	i = 0
	# for s in range(1000000):
	# 	c.move_R()
	# 	c.move_M()
	# 	i += 1
	# 	if c.is_correct_orientation():
	# 		print(i)
	# 		break

	
	c.move_R()
	c.move_U()
	# c.move_C()

	mapping_dic = dict(list(enumerate(c.orientationN)))
	
	mapping_dic_clean = {k:v for k,v in mapping_dic.items() if k != v}


	groups = []

	groupList = []

	def create_group(value, group = None):
		if group is None:
			group = []
		if value not in [item for sublist in groups for item in sublist] and value not in group:
			group.append(value)
			create_group(mapping_dic[value], group)
		else:
			if group: # Don't add any empty lists
				return groups.append(group)

	for i in mapping_dic_clean: # mapping_dic_clean for no singularities, mapping_dic for singularities included.
		create_group(i)


	print(groups)
	print(mapping_dic_clean)
	c.render(10)