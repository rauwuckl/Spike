#ifndef SPIKINGSYNAPSES_H
#define SPIKINGSYNAPSES_H

#include "Synapses.h"
#include "../Neurons/SpikingNeurons.h"


struct spiking_synapse_parameters_struct : synapse_parameters_struct {
	spiking_synapse_parameters_struct(): stdp_on(true) { synapse_parameters_struct(); }

	bool stdp_on;
	float delay_range[2];

};

class SpikingSynapses : public Synapses {

public:

	// Constructor/Destructor
	SpikingSynapses();
	~SpikingSynapses();

	// Host Pointers
	int* delays;
	bool* stdp;

	// Device pointers
	int* d_delays;
	bool* d_stdp;
	int* d_spikes_travelling_to_synapse;
	float* d_time_of_last_spike_to_reach_synapse;

	// For spike array stuff
	int maximum_axonal_delay_in_timesteps;
	

	// Synapse Functions
	virtual void AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						float timestep,
						synapse_parameters_struct * synapse_params);

	virtual void allocate_device_pointers();
	virtual void copy_constants_and_initial_efficacies_to_device();
	
	virtual void reset_synapse_activities();
	virtual void set_threads_per_block_and_blocks_per_grid(int threads);
	virtual void increment_number_of_synapses(int increment);
	virtual void shuffle_synapses();

	virtual void update_synaptic_conductances(float timestep, float current_time_in_seconds);
	virtual void calculate_postsynaptic_current_injection_components(SpikingNeurons * neurons, float current_time_in_seconds, float timestep);
	virtual void test_calcuate_total_current_injections_synapses_version_kernal(Neurons * neurons);

	virtual void interact_spikes_with_synapses(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep);

	void reset_time_related_synapse_activities();

};

// __global__ void calcuate_total_current_injections_synapses_version_kernal(float* d_component_current_injections_for_each_synapse,
// 														int* d_postsynaptic_neuron_start_indices_for_sorted_conductance_calculations,
// 														int total_number_of_synapses,
// 														float* d_current_injections,
// 														int* d_postsynaptic_neuron_indices,
// 														int total_number_of_neurons,
// 														int temp_iteration_index);

__global__ void calcuate_total_current_injections_synapses_version_kernal4(int number_of_addition_stages,
																		int* d_array_of_stage_start_indices,
																		int* d_array_of_number_of_additions_per_stage,
																		int* d_array_of_sorted_synapse_indices_for_lhs_of_addition,
																		int* d_array_of_sorted_synapse_indices_for_rhs_of_addition,
																		float* d_component_current_injections_for_each_synapse);


__global__ void calcuate_total_current_injections_synapses_version_kernal3(int start_index_for_stage,
																		int number_of_additions_for_stage,
																		int* d_array_of_sorted_synapse_indices_for_lhs_of_addition,
																		int* d_array_of_sorted_synapse_indices_for_rhs_of_addition,
																		float* d_component_current_injections_for_each_synapse);

__global__ void copy_calculated_current_injections_to_neuron_current_injection_array(int total_number_of_neurons,
																					float* d_current_injections,
																					int* d_postsynaptic_neuron_start_indices_for_sorted_conductance_calculations,
																					float* d_component_current_injections_for_each_synapse);

__global__ void calcuate_total_current_injections_synapses_version_kernal2(float* d_component_current_injections_for_each_synapse,
														int* d_postsynaptic_neuron_start_indices_for_sorted_conductance_calculations,
														int total_number_of_synapses,
														float* d_current_injections,
														int* d_postsynaptic_neuron_indices,
														int total_number_of_neurons,
														int temp_iteration_index,
														int* d_indices_of_sorted_synapses_in_orginal_arrays,
														int* d_per_neuron_afferent_synapse_count,
														int iteration_buffer);


__global__ void move_spikes_towards_synapses_kernel(int* d_presynaptic_neuron_indices,
								int* d_delays,
								int* d_spikes_travelling_to_synapse,
								float* d_neurons_last_spike_time,
								float* d_input_neurons_last_spike_time,
								float currtime,
								size_t total_number_of_synapses,
								float* d_time_of_last_spike_to_reach_synapse);

__global__ void check_bitarray_for_presynaptic_neuron_spikes(int* d_presynaptic_neuron_indices,
								int* d_delays,
								unsigned char* d_bitarray_of_neuron_spikes,
								unsigned char* d_input_neuruon_bitarray_of_neuron_spikes,
								int bitarray_length,
								int bitarray_maximum_axonal_delay_in_timesteps,
								float current_time_in_seconds,
								float timestep,
								size_t total_number_of_synapses,
								float* d_time_of_last_spike_to_reach_synapse);

#endif
