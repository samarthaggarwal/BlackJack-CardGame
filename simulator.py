import numpy as np
import ipdb

np.random.seed(32)

def get_card():
	color = "BLACK" if np.random.rand(1)[0] < 2/3 else "RED"
	card = int(np.random.rand(1)[0] * 10) + 1
	if color == "RED":
		card *= -1
	# print("card=",card)
	return card

class State:
	# initial state is the state before drawing any card
	def __init__(self):
		self.category = "GENERAL" # "SUM31", "GENERAL", "BUST"
		self.raw_sum = 0 # range = [-30,30]
		
		# whether there a black coloured 1 or not
		self.black1 = False
		self.black2 = False
		self.black3 = False
		self.opponent = 1 # TODO : remove after removing checks
		
		return

		# self.category = "GENERAL"
		# self.sum = 0
		# # whether there a black coloured 1 or not
		# self.black1 = False
		# self.black2 = False
		# self.black3 = False
		# # whether a black coloured 1 is treated as speacial or not
		# self.used1 = False
		# self.used2 = False
		# self.used3 = False

		# if card < 0:
		# 	self.category = "BUST"
		# else:
		# 	self.category = "GENERAL" # possible categories = GENERAL, BUST, 31
		# 	self.sum = card
		# 	self.black1 = card==1
		# 	self.black2 = card==2
		# 	self.black3 = card==3

	# def update_state(self, card):
		if self.sum < 0 or self.sum > 31:
			raise Exception("error, updating an end state")

		self.sum += card		
		if card < 0:
			# use atmost one unused 1,2,3
			if self.sum + 10 <=31:
				if self.black1 and not self.used1:
					self.sum += 10
					self.used1 = True
				elif self.black2 and not self.used2:
					self.sum += 10
					self.used2 = True
				elif self.black3 and not self.used3:
					self.sum += 10
					self.used3 = True

		elif card >3:
			# sum may exceed 31 so we may need to disable use of atmost 1 used 1,2,3
			if sum > 31:
				if self.used1:
					self.sum -= 10
					self.used1 = False
				elif self.used2:
					self.sum -= 10
					self.used2 = False
				elif self.used3:
					self.sum -= 10
					self.used3 = False

		else:
			# update black1, used1. may use the special card being updated
			self.black1 = self.black1 or card==1
			self.black2 = self.black2 or card==2
			self.black3 = self.black3 or card==3

			# only one of 1,2,3 would have become usable so we can use if-else 
			if self.sum + 10 <= 31:
				if self.black1 == 1 and self.used1 == False:
					self.sum += 10
					self.used1 = True
				elif self.black2 == 1 and self.used2 == False:
					self.sum += 10
					self.used2 = True
				elif self.black3 == 1 and self.used3 == False:
					self.sum += 10
					self.used3 = True

		if self.sum < 0 or self.sum > 31:
			self.category = "BUST"
		elif self.sum == 31:
			self.category = "31"
		return

	def trump_count(self):
		return int(self.black1) + int(self.black2) + int(self.black3)

	def update_state(self, card):
		if self.raw_sum < -30 or self.raw_sum > 30 or self.opponent < 1 or self.opponent > 10 or self.category == "SUM31" or self.category == "BUST":
			raise Exception("error, updating an invalid/inactionable state")

		self.raw_sum += card
		self.black1 = self.black1 or card==1
		self.black2 = self.black2 or card==2
		self.black3 = self.black3 or card==3

		# check if state can be sum31 or bust and update if so
		trump_count = self.trump_count()
		if (self.raw_sum == 31) \
			or (self.raw_sum == 21 and trump_count>=1) \
			or (self.raw_sum == 11 and trump_count>=2) \
			or (self.raw_sum == 1 and trump_count>=3):
			self.category = "SUM31"

		elif (self.raw_sum<0 and trump_count<=0) \
			or (self.raw_sum<-10 and trump_count<=1) \
			or (self.raw_sum<-20 and trump_count<=2) \
			or (self.raw_sum<-30 and trump_count<=3) \
			or (self.raw_sum>31):
			self.category = "BUST"
		return

	def best_sum(self):
		if self.category == "BUST":
			return -1
		elif self.category == "SUM31":
			return 31
		else:
			sum = self.raw_sum
			if sum + 10 <= 31 and self.black1:
				sum+=10
			if sum + 10 <= 31 and self.black2:
				sum+=10
			if sum + 10 <= 31 and self.black3:
				sum+=10

			return sum
	
	def print(self):
		print(self.category, self.raw_sum, self.black1, self.black2, self.black3, self.trump_count())

class Simulator:
	def __init__(self):
		self.agentState = State()
		card = get_card()
		self.agentState.update_state(card)

		# opponent's card
		card = get_card()
		if card < 0:
			# opponent is busted
			self.agentState.opponent = 0 # range = [0,10]
		else:
			self.agentState.opponent = card

	# after initialisation check if one of the players has busted
	def check_after_init(self):
		if self.agentState.category == "BUST" and self.agentState.opponent == 0:
			reward = 0
			done = 1
			curr_state = None
		elif self.agentState.category == "GENERAL" and self.agentState.opponent == 0:
			reward = 1
			done = 1
			curr_state = None
		elif self.agentState.category == "BUST" and self.agentState.opponent > 0:
			reward = -1
			done = 1
			curr_state = None
		else:
			reward = 0
			done = 0
			curr_state = self.agentState

		return curr_state, reward, done

	def reset(self):
		self.__init__()
		return self.agentState

	def step(self, action):
		# print("entering step,", "action=",action)
		# self.agentState.print()
		if action!="HIT" and action!="STICK":
			raise Exception("undefined action")

		if action == "HIT":
			# assuming that the player would get the chance to take an action only when his state is actionable
			card = get_card()
			self.agentState.update_state(card)
			
			# if resultant state is actionable, then return rewards, else do same as STICK action
			if self.agentState.category == "GENERAL":
				reward = 0
				done = False
		
				# print("exiting step", "reward=",reward, "done=",done)
				# self.agentState.print()
				return self.agentState, reward, done

		# play dealer's game
		if self.agentState.opponent == 0:
			dealer_category = "BUST"
		else:
			dealer_category = "GENERAL"
			self.dealerState = State()
			self.dealerState.update_state(self.agentState.opponent)

			dealer_best_sum = self.agentState.opponent
			while dealer_best_sum < 25:
				card = get_card()
				self.dealerState.update_state(card)
				dealer_best_sum = self.dealerState.best_sum()
				dealer_category = self.dealerState.category
				if dealer_category == "BUST" or dealer_category == "SUM31":
					break

		# decide the winner
		next_state = None
		done = True #done indicates end of episode
		if dealer_category == "BUST" and self.agentState.category == "BUST":
			reward = 0
		elif dealer_category == "GENERAL" and self.agentState.category == "BUST":
			reward = -1
		elif dealer_category == "BUST" and self.agentState.category == "GENERAL":
			reward = 1
		else:
			agent_best_sum = self.agentState.best_sum()
			dealer_best_sum = self.dealerState.best_sum()
			if agent_best_sum > dealer_best_sum:
				reward = 1
			elif agent_best_sum == dealer_best_sum:
				reward = 0
			else:
				reward = -1

		# print("exiting step", "reward=",reward, "done=",done)
		# self.agentState.print()
		return next_state, reward, done



