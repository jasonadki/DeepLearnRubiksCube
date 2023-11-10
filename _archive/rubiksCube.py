import numpy as np
from PIL import Image
import cv2
import random



class Cube:
	def __init__(self):
		self.orientation = [
						'O', 'O', 'O', 'O',
						'B', 'B', 'Y', 'Y',
						'G', 'G', 'B', 'B',
						'Y', 'Y', 'G', 'G',
						'R', 'R', 'R', 'R',
						'W', 'W', 'W', 'W'
					   ]

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

		self.node_orientation = [[0 for i in range(6)] for j in range(24)]
		self.update_node_orientation()


	def update_node_orientation(self):

		self.node_orientation = [[0 for i in range(6)] for j in range(24)]

		for i in enumerate(self.orientation):
			color_position = self.color_node_dic[i[1]]
			self.node_orientation[i[0]][color_position] = 1

	def render(self):
		img = self.get_image()
		img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
		print(img)
		cv2.imshow("image", np.array(img))  # show it!
		cv2.waitKey(4000)

	def get_image(self):
		env = np.zeros((8, 6, 3), dtype=np.uint8)  # starts an rbg of our size
		env[0][2] = self.colors[self.orientation[0]]
		env[0][3] = self.colors[self.orientation[1]]
		env[1][2] = self.colors[self.orientation[2]]
		env[1][3] = self.colors[self.orientation[3]]
		env[2][0] = self.colors[self.orientation[4]]
		env[2][1] = self.colors[self.orientation[5]]
		env[2][2] = self.colors[self.orientation[6]]
		env[2][3] = self.colors[self.orientation[7]]
		env[2][4] = self.colors[self.orientation[8]]
		env[2][5] = self.colors[self.orientation[9]]
		env[3][0] = self.colors[self.orientation[10]]
		env[3][1] = self.colors[self.orientation[11]]
		env[3][2] = self.colors[self.orientation[12]]
		env[3][3] = self.colors[self.orientation[13]]
		env[3][4] = self.colors[self.orientation[14]]
		env[3][5] = self.colors[self.orientation[15]]
		env[4][2] = self.colors[self.orientation[16]]
		env[4][3] = self.colors[self.orientation[17]]
		env[5][2] = self.colors[self.orientation[18]]
		env[5][3] = self.colors[self.orientation[19]]
		env[6][2] = self.colors[self.orientation[20]]
		env[6][3] = self.colors[self.orientation[21]]
		env[7][2] = self.colors[self.orientation[22]]
		env[7][3] = self.colors[self.orientation[23]]

		img = Image.fromarray(env, 'RGB')
		return img

	def swap(self, lst, pos1, pos2):
		"""
		Swap the elements in a list based off position.
		
		lst <list>: the list that needs contents switched
		pos1 <int>: the first position that needs to be swaped
		pos2 <int>: the second position that needs to be swaped

		:return: <list> the newly modified list.
		"""

		a, b = lst[pos1], lst[pos2]
		lst[pos1] = b
		lst[pos2] = a

		return lst

	def move_R(self):
		a = self.orientation[1]
		self.orientation[1] = self.orientation[3]
		self.orientation[3] = self.orientation[7]
		self.orientation[7] = self.orientation[13]
		self.orientation[13] = self.orientation[17]
		self.orientation[17] = self.orientation[19]
		self.orientation[19] = self.orientation[21]
		self.orientation[21] = self.orientation[23]
		self.orientation[23] = a

		self.update_node_orientation()

	def move_Rp(self):
		a = self.orientation[1]
		self.orientation[1] = self.orientation[23]
		self.orientation[23] = self.orientation[21]
		self.orientation[21] = self.orientation[19]
		self.orientation[19] = self.orientation[17]
		self.orientation[17] = self.orientation[13]
		self.orientation[13] = self.orientation[7]
		self.orientation[7] = self.orientation[3]
		self.orientation[3] = a

		self.update_node_orientation()

	def move_L(self):
		a = self.orientation[0]
		self.orientation[0] = self.orientation[2]
		self.orientation[2] = self.orientation[6]
		self.orientation[6] = self.orientation[12]
		self.orientation[12] = self.orientation[16]
		self.orientation[16] = self.orientation[18]
		self.orientation[18] = self.orientation[20]
		self.orientation[20] = self.orientation[22]
		self.orientation[22] = a

		self.update_node_orientation()

	def move_Lp(self):
		a = self.orientation[0]
		self.orientation[0] = self.orientation[22]
		self.orientation[22] = self.orientation[20]
		self.orientation[20] = self.orientation[18]
		self.orientation[18] = self.orientation[16]
		self.orientation[16] = self.orientation[12]
		self.orientation[12] = self.orientation[6]
		self.orientation[6] = self.orientation[2]
		self.orientation[2] = a

		self.update_node_orientation()

	def move_U(self):

		a = self.orientation[6]
		self.orientation[6] = self.orientation[7]
		self.orientation[7] = self.orientation[13]
		self.orientation[13] = self.orientation[12]
		self.orientation[12] = a

		a = self.orientation[2]
		self.orientation[2] = self.orientation[8]
		self.orientation[8] = self.orientation[17]
		self.orientation[17] = self.orientation[11]
		self.orientation[11] = a

		a = self.orientation[3]
		self.orientation[3] = self.orientation[14]
		self.orientation[14] = self.orientation[16]
		self.orientation[16] = self.orientation[5]
		self.orientation[5] = a

		self.update_node_orientation()

	def move_Up(self):

		a = self.orientation[6]
		self.orientation[6] = self.orientation[12]
		self.orientation[12] = self.orientation[13]
		self.orientation[13] = self.orientation[7]
		self.orientation[7] = a

		a = self.orientation[2]
		self.orientation[2] = self.orientation[11]
		self.orientation[11] = self.orientation[17]
		self.orientation[17] = self.orientation[8]
		self.orientation[8] = a

		a = self.orientation[3]
		self.orientation[3] = self.orientation[5]
		self.orientation[5] = self.orientation[16]
		self.orientation[16] = self.orientation[14]
		self.orientation[14] = a

		self.update_node_orientation()

	def move_B(self):

		a = self.orientation[21]
		self.orientation[21] = self.orientation[20]
		self.orientation[20] = self.orientation[22]
		self.orientation[22] = self.orientation[23]
		self.orientation[23] = a

		a = self.orientation[0]
		self.orientation[0] = self.orientation[9]
		self.orientation[9] = self.orientation[19]
		self.orientation[19] = self.orientation[10]
		self.orientation[10] = a

		a = self.orientation[1]
		self.orientation[1] = self.orientation[15]
		self.orientation[15] = self.orientation[18]
		self.orientation[18] = self.orientation[4]
		self.orientation[4] = a

		self.update_node_orientation()

	def move_Bp(self):

		a = self.orientation[21]
		self.orientation[21] = self.orientation[23]
		self.orientation[23] = self.orientation[22]
		self.orientation[22] = self.orientation[20]
		self.orientation[20] = a

		a = self.orientation[0]
		self.orientation[0] = self.orientation[10]
		self.orientation[10] = self.orientation[19]
		self.orientation[19] = self.orientation[9]
		self.orientation[9] = a

		a = self.orientation[1]
		self.orientation[1] = self.orientation[4]
		self.orientation[4] = self.orientation[18]
		self.orientation[18] = self.orientation[15]
		self.orientation[15] = a

		self.update_node_orientation()

	def move_F(self):

		a = self.orientation[17]
		self.orientation[17] = self.orientation[16]
		self.orientation[16] = self.orientation[18]
		self.orientation[18] = self.orientation[19]
		self.orientation[19] = a

		a = self.orientation[13]
		self.orientation[13] = self.orientation[11]
		self.orientation[11] = self.orientation[20]
		self.orientation[20] = self.orientation[15]
		self.orientation[15] = a

		a = self.orientation[12]
		self.orientation[12] = self.orientation[10]
		self.orientation[10] = self.orientation[21]
		self.orientation[21] = self.orientation[14]
		self.orientation[14] = a

		self.update_node_orientation()

	def move_Fp(self):

		a = self.orientation[17]
		self.orientation[17] = self.orientation[19]
		self.orientation[19] = self.orientation[18]
		self.orientation[18] = self.orientation[16]
		self.orientation[16] = a

		a = self.orientation[13]
		self.orientation[13] = self.orientation[15]
		self.orientation[15] = self.orientation[20]
		self.orientation[20] = self.orientation[11]
		self.orientation[11] = a

		a = self.orientation[12]
		self.orientation[12] = self.orientation[14]
		self.orientation[14] = self.orientation[21]
		self.orientation[21] = self.orientation[10]
		self.orientation[10] = a

		self.update_node_orientation()

	def move_O(self):

		a = self.orientation[3]
		self.orientation[3] = self.orientation[2]
		self.orientation[2] = self.orientation[0]
		self.orientation[0] = self.orientation[1]
		self.orientation[1] = a

		a = self.orientation[7]
		self.orientation[7] = self.orientation[5]
		self.orientation[5] = self.orientation[22]
		self.orientation[22] = self.orientation[9]
		self.orientation[9] = a

		a = self.orientation[6]
		self.orientation[6] = self.orientation[4]
		self.orientation[4] = self.orientation[23]
		self.orientation[23] = self.orientation[8]
		self.orientation[8] = a

		self.update_node_orientation()

	def move_Op(self):

		a = self.orientation[3]
		self.orientation[3] = self.orientation[1]
		self.orientation[1] = self.orientation[0]
		self.orientation[0] = self.orientation[2]
		self.orientation[2] = a

		a = self.orientation[7]
		self.orientation[7] = self.orientation[9]
		self.orientation[9] = self.orientation[22]
		self.orientation[22] = self.orientation[5]
		self.orientation[5] = a

		a = self.orientation[6]
		self.orientation[6] = self.orientation[8]
		self.orientation[8] = self.orientation[23]
		self.orientation[23] = self.orientation[4]
		self.orientation[4] = a

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


