import numpy as np
from PIL import Image
import cv2
import random
from random import randrange
import time
import copy

class Cube:
	def __init__(self):

		

		self.done = True
		self.reward = 0

		self.action_space = 13 # 12 moves + 1 do nothing
		self.state_space = 144 # 24 stickers * 6 colors each


		# Initial Orientation
		self.initOrientation = [
			[0,0,0,1,0,0],
			[0,0,0,1,0,0],
			[0,0,0,1,0,0],
			[0,0,0,1,0,0],
			[1,0,0,0,0,0],
			[1,0,0,0,0,0],
			[0,0,1,0,0,0],
			[0,0,1,0,0,0],
			[0,0,0,0,1,0],
			[0,0,0,0,1,0],
			[1,0,0,0,0,0],
			[1,0,0,0,0,0],
			[0,0,1,0,0,0],
			[0,0,1,0,0,0],
			[0,0,0,0,1,0],
			[0,0,0,0,1,0],
			[0,1,0,0,0,0],
			[0,1,0,0,0,0],
			[0,1,0,0,0,0],
			[0,1,0,0,0,0],
			[0,0,0,0,0,1],
			[0,0,0,0,0,1],
			[0,0,0,0,0,1],
			[0,0,0,0,0,1]]

		self.orientation = copy.deepcopy(self.initOrientation)

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
		:return: <bool> Whether the cube is solved
		"""
		
		# Define the face groups with corrected indices
		f1 = [0, 1, 2, 3]  # Back
		f2 = [4, 5, 10, 11]  # Left
		f3 = [6, 7, 12, 13]  # Top
		f4 = [8, 9, 14, 15]  # Right
		f5 = [16, 17, 18, 19]  # Front
		f6 = [20, 21, 22, 23]  # Bottom
		
		# Iterate through each list and check whether all are the same color
		for face in [f1, f2, f3, f4, f5, f6]:
			# Get the position of the '1' for each square in the face
			pos = [self.orientation[s].index(1) for s in face]
			# If not all positions (colors) are the same for the face, return False
			if pos.count(pos[0]) != len(pos):
				self.done = False
				return False


		self.done = True

		return True



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
		self.orientation_flat = np.array(self.orientation).flatten().tolist()


	def render(self, timeDisplay = 2):

		img = self.get_image()
		img = img.resize((300, 300), resample = Image.NEAREST)  # resizing so we can see our agent in all its glory.
		
		cv2.imshow("image", np.array(img))  # show it!
		cv2.waitKey(timeDisplay * 1000)
		cv2.destroyAllWindows()


	def get_image(self):
		
		env = np.zeros((8, 6, 3), dtype=np.uint8)  # starts an rbg of our size
		env[0][2] = self.colors[self.orientation[0].index(1)]
		env[0][3] = self.colors[self.orientation[1].index(1)]
		env[1][2] = self.colors[self.orientation[2].index(1)]
		env[1][3] = self.colors[self.orientation[3].index(1)]
		env[2][0] = self.colors[self.orientation[4].index(1)]
		env[2][1] = self.colors[self.orientation[5].index(1)]
		env[2][2] = self.colors[self.orientation[6].index(1)]
		env[2][3] = self.colors[self.orientation[7].index(1)]
		env[2][4] = self.colors[self.orientation[8].index(1)]
		env[2][5] = self.colors[self.orientation[9].index(1)]
		env[3][0] = self.colors[self.orientation[10].index(1)]
		env[3][1] = self.colors[self.orientation[11].index(1)]
		env[3][2] = self.colors[self.orientation[12].index(1)]
		env[3][3] = self.colors[self.orientation[13].index(1)]
		env[3][4] = self.colors[self.orientation[14].index(1)]
		env[3][5] = self.colors[self.orientation[15].index(1)]
		env[4][2] = self.colors[self.orientation[16].index(1)]
		env[4][3] = self.colors[self.orientation[17].index(1)]
		env[5][2] = self.colors[self.orientation[18].index(1)]
		env[5][3] = self.colors[self.orientation[19].index(1)]
		env[6][2] = self.colors[self.orientation[20].index(1)]
		env[6][3] = self.colors[self.orientation[21].index(1)]
		env[7][2] = self.colors[self.orientation[22].index(1)]
		env[7][3] = self.colors[self.orientation[23].index(1)]


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
		| ^
		| |
		| |
		"""
		self.shift_orientation([7, 17, 21, 1])
		self.shift_orientation([13, 19, 23, 3])

		self.shift_orientation([14, 15, 9, 8])

		self.update_flat_orientation()
		

		self.is_correct_orientation()

	def move_Rp(self):
		"""
		| |
		| |
		| v
		"""
		
		self.shift_orientation([7, 1, 21, 17])
		self.shift_orientation([13, 3, 23, 19])

		self.shift_orientation([14, 8, 9, 15])

		self.update_flat_orientation()
		
		self.is_correct_orientation()

	def move_L(self):
		"""
		^ |
		| |
		| |
		"""
		
		self.shift_orientation([6, 16, 20, 0])
		self.shift_orientation([12, 18, 22, 2])

		self.shift_orientation([5, 11, 10, 4])

		self.update_flat_orientation()
		
		self.is_correct_orientation()

	def move_Lp(self):
		"""
		| |
		| |
		v |
		"""
		
		
		self.shift_orientation([6, 0, 20, 16])
		self.shift_orientation([12, 2, 22, 18])

		self.shift_orientation([5, 4, 10, 11])
		self.update_flat_orientation()
		
		self.is_correct_orientation()



	def move_U(self):
		"""
		< - -
		- - -
		"""
		
		self.shift_orientation([17, 8, 2, 11])
		self.shift_orientation([16, 14, 3, 5])

		self.shift_orientation([13, 7, 6, 12])

		self.update_flat_orientation()
		
		self.is_correct_orientation()

	def move_Up(self):
		"""
		- - >
		- - -
		"""
		
		self.shift_orientation([17, 11, 2, 8])
		self.shift_orientation([16, 5, 3, 14])

		self.shift_orientation([13, 12, 6, 7])
		self.update_flat_orientation()
		
		self.is_correct_orientation()


	def move_B(self):
		"""
		- - -
		- - >
		"""
		
		self.shift_orientation([19, 10, 0, 9])
		self.shift_orientation([18, 4, 1, 15])

		self.shift_orientation([20, 22, 23, 21])

		self.update_flat_orientation()
		
		self.is_correct_orientation()

	def move_Bp(self):
		"""
		- - -
		< - -
		"""
		
		self.shift_orientation([19, 9, 0, 10])
		self.shift_orientation([18, 15, 1, 4])

		self.shift_orientation([20, 21, 23, 22])

		self.update_flat_orientation()
		
		self.is_correct_orientation()


	def move_F(self):
		"""
		- - -
		^   v
		- - -
		"""
		
		self.shift_orientation([12, 10, 21, 14])
		self.shift_orientation([13, 11, 20, 15])

		self.shift_orientation([16, 18, 19, 17])

		self.update_flat_orientation()
		
		self.is_correct_orientation()

	def move_Fp(self):
		"""
		- - -
		v   ^
		- - -
		"""
		
		self.shift_orientation([12, 14, 21, 10])
		self.shift_orientation([13, 15, 20, 11])

		self.shift_orientation([16, 17, 19, 18])

		self.update_flat_orientation()
		
		self.is_correct_orientation()

	def move_O(self):
		"""
		- 		- 	-

		v Backside  ^

		- 		- 	-
		"""
		
		self.shift_orientation([6, 8, 23, 4])
		self.shift_orientation([7, 9, 22, 5])

		self.shift_orientation([3, 0, 1, 2])

		self.update_flat_orientation()
		
		self.is_correct_orientation()

	def move_Op(self):
		"""
		- 		- 	-

		^ Backside  v

		- 		- 	-
		"""
		
		
		self.shift_orientation([6, 4, 23, 8])
		self.shift_orientation([7, 5, 22, 9])

		self.shift_orientation([3, 2, 1, 0])

		self.update_flat_orientation()
		
		self.is_correct_orientation()



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

		for _ in range(n):
			# Get the random move from 0-11
			move = randrange(12)

			if move == 0:
				self.move_R()
			elif move == 1:
				self.move_Rp()
			elif move == 2:
				self.move_L()
			elif move == 3:
				self.move_Lp()
			elif move == 4:
				self.move_U()
			elif move == 5:
				self.move_Up()
			elif move == 6:
				self.move_B()
			elif move == 7:
				self.move_Bp()
			elif move == 8:
				self.move_F()
			elif move == 9:
				self.move_Fp()
			elif move == 10:
				self.move_O()
			elif move == 11:
				self.move_Op()


		return self.orientation_flat, move


	def step(self, action):
		"""
		Make a turn on the cube given an action number
		"""

		
		self.reward = 0
		# self.done = False

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
			self.move_L()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 3:
			self.move_Lp()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 4:
			self.move_U()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 5:
			self.move_Up()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 6:
			self.move_B()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 7:
			self.move_Bp()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 8:
			self.move_F()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 9:
			self.move_Fp()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 10:
			self.move_O()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		elif action == 11:
			self.move_Op()
			if self.done:
				self.reward = 100
			else:
				self.reward -= 1
		# elif action == 12:
		# 	if self.is_correct_orientation():
		# 		self.reward = 100
		# 		self.done = True
		# 	else:
		# 		self.reward -= 1
		# 		self.done = False

		return self.reward, self.orientation_flat, self.done



if __name__ == '__main__':
	

	# Time how long each step takes to run
	for i in range(12):
		c = Cube()
		# Run the step function 3 times and average the time
		tArr = []
		for j in range(5000):
			start = time.time()
			c.step(i)
			end = time.time()
			tArr.append(end - start)
		
		# print the average time to 10 decimal places
		print("Average time for step {} is {:.10f} seconds".format(i, sum(tArr) / len(tArr)))

