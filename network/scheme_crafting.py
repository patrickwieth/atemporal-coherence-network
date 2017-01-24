import random

from network import schemes
from network import mechanisms


class scheme_dice:
	def __init__(self):	
		# exclusive_must_haves contains arrays from which one and only one item will be picked, thus every array provides exactly one mechanism
		self.exclusive_must_haves = []
		self.exclusive_must_haves.append(['weight_init1'])
		self.exclusive_must_haves.append(['input_sum1'])

		self.exclusive_must_haves.append(['con1'])
		self.exclusive_must_haves.append(['rcv1'])
		self.exclusive_must_haves.append(['brc1'])

		# inclusive_optionals contains arrays which are either included with all items or with none, thus every array provides either all or none of its mechanisms
		self.inclusive_optionals = []
		self.inclusive_optionals.append(['input_post1'])
		self.inclusive_optionals.append(['input_post2'])
		self.inclusive_optionals.append(['ac_pre1'])
		self.inclusive_optionals.append(['ac_pre2', 'ac_pro1'])
		self.inclusive_optionals.append(['ac_post1'])



	def roll_dice(self):
		optionals = [random.randint(0,1) for x in self.inclusive_optionals]
		must_haves = [random.randint(1,len(x)) for x in self.exclusive_must_haves]

		return {'optionals': optionals, 'must_haves': must_haves}

	def create_scheme(self, rolled_dice):
		new_scheme = schemes.scheme()

		selected_mechanisms = []

		for idx, val in enumerate(rolled_dice['optionals']):
			if(val == 1):
				selected_mechanisms += self.inclusive_optionals[idx]

		for idx, val in enumerate(rolled_dice['must_haves']):
			selected_mechanisms.append(self.exclusive_must_haves[idx][val-1])

		mechanism_set = {'input_processing': [], 
						 'input_postprocessing': [],
						 'activation_preprocessing': [], 
						 'activation_processing': [], 
						 'activation_postprocessing': [],
						 'broadcast': [],
						 'receive': [],
						 'connect': []
						}

		for x in selected_mechanisms:	
			picked = mechanisms.pool[x]
			mechanism_set[picked[1]] += [picked[0]] #

		print(mechanism_set)

		new_scheme.set_mechanisms_by_dict(mechanism_set)



test = scheme_dice()

rollit = test.roll_dice()

test.create_scheme(rollit)

'''
pool = {'weight_init1':	[add_weights_when_input_too_long, 'input_processing'],
		'input_sum1':	[sum_up_input, 'input_processing'],
		'input_post1':	[increase_weights_decrease_activation_on_weak_input, 'input_postprocessing'],
		'input_post2':	[define_unset_activation, 'input_postprocessing'],
		'ac_pre1':		[empty_actives, 'activation_preprocessing'],
		'ac_pre2':		[input_intensity_by_abs_diff, 'activation_preprocessing'],
		'ac_pro1':		[buff_activation_on_strong_input_nerv_on_weak_input, 'activation_processing'],
		'ac_post1':		[donothing, 'activation_postprocessing'],
		'brc1':			[broadcast_intercon, "broadcast"],
		'rcv1':			[receive_intercon, 'receive'],
		'con1':			[add_intercon, 'connect']
		}
'''
