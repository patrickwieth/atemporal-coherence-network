import random

from network import schemes
from network import mechanisms


class scheme_dice:
	def __init__(self):	
		# exclusive_must_haves contains arrays from which one and only one item will be picked, thus every array provides exactly one mechanism
		self.exclusive_must_haves = []
		self.exclusive_must_haves.append([mechanisms.add_weights_when_input_too_long])
		self.exclusive_must_haves.append([mechanisms.sum_up_input])
		self.exclusive_must_haves.append([mechanisms.add_intercon])
		self.exclusive_must_haves.append([mechanisms.receive_intercon])
		self.exclusive_must_haves.append([mechanisms.broadcast_intercon])

		# inclusive_optionals contains arrays which are either included with all items or with none, thus every array provides either all or none of its mechanisms
		self.inclusive_optionals = []
		self.inclusive_optionals.append([mechanisms.increase_weights_decrease_activation_on_weak_input])
		self.inclusive_optionals.append([mechanisms.define_unset_activation])
		self.inclusive_optionals.append([mechanisms.empty_actives])
		self.inclusive_optionals.append([mechanisms.input_intensity_by_abs_diff, mechanisms.buff_activation_on_strong_input_nerf_on_weak_input])
		self.inclusive_optionals.append([mechanisms.input_intensity_by_abs_diff, mechanisms.scale_weights_on_strong_input_scale_down_activation_on_weak_input])
		self.inclusive_optionals.append([mechanisms.do_nothing])


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

		new_scheme.set_mechanisms_by_list(selected_mechanisms)

		return new_scheme
