#include "SpikingNeurons.hpp"
#include <stdlib.h>

// SpikingNeurons Constructor
SpikingNeurons::SpikingNeurons() {
	// Variables
	bitarray_length = 0;
	bitarray_maximum_axonal_delay_in_timesteps = 0;
	high_fidelity_spike_flag = false;

	// Host Pointers
	after_spike_reset_membrane_potentials_c = NULL;
	thresholds_for_action_potential_spikes = NULL;
	bitarray_of_neuron_spikes = NULL;
}

// SpikingNeurons Destructor
SpikingNeurons::~SpikingNeurons() {
	free(after_spike_reset_membrane_potentials_c);
	free(thresholds_for_action_potential_spikes);
	free(bitarray_of_neuron_spikes);
}

int SpikingNeurons::AddGroup(neuron_parameters_struct * group_params){
	int new_group_id = Neurons::AddGroup(group_params);

	spiking_neuron_parameters_struct * spiking_group_params = (spiking_neuron_parameters_struct*)group_params;

	after_spike_reset_membrane_potentials_c = (float*)realloc(after_spike_reset_membrane_potentials_c, (total_number_of_neurons*sizeof(float)));
	thresholds_for_action_potential_spikes = (float*)realloc(thresholds_for_action_potential_spikes, (total_number_of_neurons*sizeof(float)));

	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
		after_spike_reset_membrane_potentials_c[i] = spiking_group_params->resting_potential_v0;
		thresholds_for_action_potential_spikes[i] = spiking_group_params->threshold_for_action_potential_spike;
	}

	return new_group_id;
}

#include <iostream>
void SpikingNeurons::reset_state() {
  std::cout << backend << "!\n";
  Neurons::reset_state();
  backend->reset_state();
}


void SpikingNeurons::update_membrane_potentials(float timestep, float current_time_in_seconds) {
	
}

void SpikingNeurons::check_for_neuron_spikes(float current_time_in_seconds, float timestep) {
  backend->check_for_neuron_spikes(current_time_in_seconds, timestep);
  // TODO: Copy the result from the backend
}

MAKE_STUB_PREPARE_BACKEND(SpikingNeurons);
