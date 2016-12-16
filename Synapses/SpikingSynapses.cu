#include "SpikingSynapses.h"

#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"


// SpikingSynapses Constructor
SpikingSynapses::SpikingSynapses() {

	delays = NULL;
	stdp = NULL;

	d_delays = NULL;
	d_spikes_travelling_to_synapse = NULL;
	d_stdp = NULL;
	d_time_of_last_spike_to_reach_synapse = NULL;

	maximum_axonal_delay_in_timesteps = 0;
}

// SpikingSynapses Destructor
SpikingSynapses::~SpikingSynapses() {
	// Just need to free up the memory
	// Full Matrices
	free(delays);
	free(stdp);

	CudaSafeCall(cudaFree(d_delays));
	CudaSafeCall(cudaFree(d_spikes_travelling_to_synapse));
	CudaSafeCall(cudaFree(d_stdp));
	CudaSafeCall(cudaFree(d_time_of_last_spike_to_reach_synapse));

}

// Connection Detail implementation
//	INPUT:
//		Pre-neuron population ID
//		Post-neuron population ID
//		An array of the exclusive sum of neuron populations
//		CONNECTIVITY_TYPE (Constants.h)
//		2 number float array for weight range
//		2 number float array for delay range
//		Boolean value to indicate if population is STDP based
//		Parameter = either probability for random synapses or S.D. for Gaussian
void SpikingSynapses::AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						float timestep,
						synapse_parameters_struct * synapse_params) {
	
	
	Synapses::AddGroup(presynaptic_group_id, 
							postsynaptic_group_id, 
							neurons,
							input_neurons,
							timestep,
							synapse_params);

	spiking_synapse_parameters_struct * spiking_synapse_group_params = (spiking_synapse_parameters_struct*)synapse_params;

	for (int i = original_number_of_synapses; i < total_number_of_synapses; i++){
		
		// Convert delay range from time to number of timesteps
		int delay_range_in_timesteps[2] = {int(round(spiking_synapse_group_params->delay_range[0]/timestep)), int(round(spiking_synapse_group_params->delay_range[1]/timestep))};

		// Check delay range bounds greater than timestep
		if ((delay_range_in_timesteps[0] < 1) || (delay_range_in_timesteps[1] < 1)) {
			printf("%d\n", delay_range_in_timesteps[0]);
			printf("%d\n", delay_range_in_timesteps[1]);
			print_message_and_exit("Delay range must be at least one timestep.");
		}

		// Setup Delays
		if (delay_range_in_timesteps[0] == delay_range_in_timesteps[1]) {
			delays[i] = delay_range_in_timesteps[0];
		} else {
			float random_delay = delay_range_in_timesteps[0] + (delay_range_in_timesteps[1] - delay_range_in_timesteps[0]) * ((float)rand() / (RAND_MAX));
			delays[i] = round(random_delay);
		}

		// printf("delay_range_in_timesteps[0]: %d\n", delay_range_in_timesteps[0]);
		// printf("delay_range_in_timesteps[1]: %d\n", delay_range_in_timesteps[1]);

		if (delay_range_in_timesteps[0] > maximum_axonal_delay_in_timesteps){
			maximum_axonal_delay_in_timesteps = delay_range_in_timesteps[0];
		} else if (delay_range_in_timesteps[1] > maximum_axonal_delay_in_timesteps){
			maximum_axonal_delay_in_timesteps = delay_range_in_timesteps[1];
		}

		// printf("maximum_axonal_delay_in_timesteps: %d\n", maximum_axonal_delay_in_timesteps);

		//Set STDP on or off for synapse
		stdp[i] = spiking_synapse_group_params->stdp_on;
	}

}

void SpikingSynapses::increment_number_of_synapses(int increment) {

	Synapses::increment_number_of_synapses(increment);

    delays = (int*)realloc(delays, total_number_of_synapses * sizeof(int));
    stdp = (bool*)realloc(stdp, total_number_of_synapses * sizeof(bool));

}


void SpikingSynapses::allocate_device_pointers() {

	Synapses::allocate_device_pointers();

	CudaSafeCall(cudaMalloc((void **)&d_delays, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_stdp, sizeof(bool)*total_number_of_synapses));

	CudaSafeCall(cudaMalloc((void **)&d_spikes_travelling_to_synapse, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_time_of_last_spike_to_reach_synapse, sizeof(float)*total_number_of_synapses));

	
}


void SpikingSynapses::copy_constants_and_initial_efficacies_to_device() {
	
	Synapses::copy_constants_and_initial_efficacies_to_device();

	CudaSafeCall(cudaMemcpy(d_delays, delays, sizeof(int)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_stdp, stdp, sizeof(bool)*total_number_of_synapses, cudaMemcpyHostToDevice));

}


void SpikingSynapses::reset_synapse_activities() {

	Synapses::reset_synapse_activities();
	
	reset_time_related_synapse_activities();

}

void SpikingSynapses::reset_time_related_synapse_activities() {

	CudaSafeCall(cudaMemset(d_spikes_travelling_to_synapse, 0, sizeof(int)*total_number_of_synapses));
	// Set last spike times to -1000 so that the times do not affect current simulation.
	float* last_spike_to_reach_synapse;
	last_spike_to_reach_synapse = (float*)malloc(sizeof(float)*total_number_of_synapses);
	for (int i=0; i < total_number_of_synapses; i++){
		last_spike_to_reach_synapse[i] = -1000.0f;
	}
	CudaSafeCall(cudaMemcpy(d_time_of_last_spike_to_reach_synapse, last_spike_to_reach_synapse, total_number_of_synapses*sizeof(float), cudaMemcpyHostToDevice));

}


void SpikingSynapses::shuffle_synapses() {
	
	Synapses::shuffle_synapses();

	int * temp_delays = (int *)malloc(total_number_of_synapses*sizeof(int));
	bool * temp_stdp = (bool *)malloc(total_number_of_synapses*sizeof(bool));
	for(int i = 0; i < total_number_of_synapses; i++) {

		temp_delays[i] = delays[original_synapse_indices[i]];
		temp_stdp[i] = stdp[original_synapse_indices[i]];

	}

	delays = temp_delays;
	stdp = temp_stdp;

}


void SpikingSynapses::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	Synapses::set_threads_per_block_and_blocks_per_grid(threads);
	
}

void SpikingSynapses::interact_spikes_with_synapses(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) {

	if (neurons->high_fidelity_spike_flag){
		check_bitarray_for_presynaptic_neuron_spikes<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(
								d_presynaptic_neuron_indices,
								d_delays,
								neurons->d_bitarray_of_neuron_spikes,
								input_neurons->d_bitarray_of_neuron_spikes,
								neurons->bitarray_length,
								neurons->bitarray_maximum_axonal_delay_in_timesteps,
								current_time_in_seconds,
								timestep,
								total_number_of_synapses,
								d_time_of_last_spike_to_reach_synapse);
		CudaCheckError();
	}
	else{
		move_spikes_towards_synapses_kernel<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_presynaptic_neuron_indices,
																			d_delays,
																			d_spikes_travelling_to_synapse,
																			neurons->d_last_spike_time_of_each_neuron,
																			input_neurons->d_last_spike_time_of_each_neuron,
																			current_time_in_seconds,
																			total_number_of_synapses,
																			d_time_of_last_spike_to_reach_synapse);
		CudaCheckError();
	}
}



void SpikingSynapses::calculate_postsynaptic_current_injection_components(SpikingNeurons * neurons, float current_time_in_seconds, float timestep) {

}

void SpikingSynapses::update_synaptic_conductances(float timestep, float current_time_in_seconds) {

}


void SpikingSynapses::test_calcuate_total_current_injections_synapses_version_kernal(Neurons * neurons) {


calcuate_total_current_injections_synapses_version_kernal4<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(number_of_addition_stages,
																															d_array_of_stage_start_indices,
																															d_array_of_number_of_additions_per_stage,
																															d_array_of_sorted_synapse_indices_for_lhs_of_addition,
																															d_array_of_sorted_synapse_indices_for_rhs_of_addition,
																															d_component_current_injections_for_each_synapse);


	// for (int stage_count = 0; stage_count < number_of_addition_stages; stage_count++) {

	// for (int stage_count = 0; stage_count < 1; stage_count++) {

	// 	calcuate_total_current_injections_synapses_version_kernal3<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(array_of_stage_start_indices[stage_count],
	// 																														array_of_number_of_additions_per_stage[stage_count],
	// 																														d_array_of_sorted_synapse_indices_for_lhs_of_addition,
	// 																														d_array_of_sorted_synapse_indices_for_rhs_of_addition,
	// 																														d_component_current_injections_for_each_synapse);

	// }

	CudaCheckError();


	copy_calculated_current_injections_to_neuron_current_injection_array<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(neurons->total_number_of_neurons,
																			neurons->d_current_injections,
																			neurons->d_postsynaptic_neuron_start_indices_for_sorted_conductance_calculations,
																			d_component_current_injections_for_each_synapse);

	CudaCheckError();


}

__global__ void calcuate_total_current_injections_synapses_version_kernal4(int number_of_addition_stages,
																		int* d_array_of_stage_start_indices,
																		int* d_array_of_number_of_additions_per_stage,
																		int* d_array_of_sorted_synapse_indices_for_lhs_of_addition,
																		int* d_array_of_sorted_synapse_indices_for_rhs_of_addition,
																		float* d_component_current_injections_for_each_synapse) {


	// int start_index_for_stage = d_array_of_stage_start_indices[stage_count];
	int start_index_for_stage = 0;

	for (int stage_count = 0; stage_count < number_of_addition_stages; stage_count++) {
		int number_of_additions_for_stage = d_array_of_number_of_additions_per_stage[stage_count];
		if (stage_count!= 0) {
			start_index_for_stage += number_of_addition_stages;	
		}
		

		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		while (idx < number_of_additions_for_stage) {

			d_component_current_injections_for_each_synapse[d_array_of_sorted_synapse_indices_for_lhs_of_addition[start_index_for_stage+idx]] += d_component_current_injections_for_each_synapse[d_array_of_sorted_synapse_indices_for_rhs_of_addition[start_index_for_stage+idx]];

			idx += blockDim.x * gridDim.x;
		
		}
	// __syncthreads();

	}


}



// __global__ void calcuate_total_current_injections_synapses_version_kernal3(int start_index_for_stage,
// 																		int number_of_additions_for_stage,
// 																		int* d_array_of_sorted_synapse_indices_for_lhs_of_addition,
// 																		int* d_array_of_sorted_synapse_indices_for_rhs_of_addition,
// 																		float* d_component_current_injections_for_each_synapse) {


// 	for (int stage_count = 0; stage_count < 17; stage_count++) {

// 		int idx = threadIdx.x + blockIdx.x * blockDim.x;
// 		while (idx < number_of_additions_for_stage) {

// 			d_component_current_injections_for_each_synapse[d_array_of_sorted_synapse_indices_for_lhs_of_addition[start_index_for_stage+idx]] += d_component_current_injections_for_each_synapse[d_array_of_sorted_synapse_indices_for_rhs_of_addition[start_index_for_stage+idx]];

// 			idx += blockDim.x * gridDim.x;
		
// 		}
// 	__syncthreads();

// 	}


// }


__global__ void copy_calculated_current_injections_to_neuron_current_injection_array(int total_number_of_neurons,
																					float* d_current_injections,
																					int* d_postsynaptic_neuron_start_indices_for_sorted_conductance_calculations,
																					float* d_component_current_injections_for_each_synapse) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_neurons) {

		d_current_injections[idx] = d_component_current_injections_for_each_synapse[d_postsynaptic_neuron_start_indices_for_sorted_conductance_calculations[idx]];

		idx += blockDim.x * gridDim.x;
	}
	__syncthreads();


}




// void SpikingSynapses::test_calcuate_total_current_injections_synapses_version_kernal(Neurons * neurons) {


// 	for (int temp_iteration_index = 0; temp_iteration_index < 5; temp_iteration_index++) {

// 		int iteration_buffer = powf(temp_iteration_index, 2);

// 		// printf("temp_iteration_index: %d\n", temp_iteration_index);
// 		// calcuate_total_current_injections_synapses_version_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_component_current_injections_for_each_synapse,
// 		// 												neurons->d_postsynaptic_neuron_start_indices_for_sorted_conductance_calculations,
// 		// 												total_number_of_synapses,
// 		// 												neurons->d_current_injections,
// 		// 												d_postsynaptic_neuron_indices,
// 		// 												neurons->total_number_of_neurons,
// 		// 												temp_iteration_index);

// 		calcuate_total_current_injections_synapses_version_kernal2<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_component_current_injections_for_each_synapse,
// 														neurons->d_postsynaptic_neuron_start_indices_for_sorted_conductance_calculations,
// 														total_number_of_synapses,
// 														neurons->d_current_injections,
// 														d_postsynaptic_neuron_indices,
// 														neurons->total_number_of_neurons,
// 														temp_iteration_index,
// 														d_indices_of_sorted_synapses_in_orginal_arrays,
// 														neurons->d_per_neuron_afferent_synapse_count,
// 														iteration_buffer);

// 		CudaCheckError();

// 	}

// 	// printf("end\n");

// }


// __global__ void calcuate_total_current_injections_synapses_version_kernal2(float* d_component_current_injections_for_each_synapse,
// 														int* d_postsynaptic_neuron_start_indices_for_sorted_conductance_calculations,
// 														int total_number_of_synapses,
// 														float* d_current_injections,
// 														int* d_postsynaptic_neuron_indices,
// 														int total_number_of_neurons,
// 														int temp_iteration_index,
// 														int* d_indices_of_sorted_synapses_in_orginal_arrays,
// 														int* d_per_neuron_afferent_synapse_count,
// 														int iteration_buffer) {

// 	// Get thread IDs
// 	int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
// 	int idx = t_idx;

// 	temp_iteration_index = 0;

// 	total_number_of_synapses = total_number_of_synapses / 2;

// 	while (idx < total_number_of_synapses) {

// 		int sorted_syn1_idx = idx;
// 		int sorted_syn2_idx = sorted_syn1_idx + iteration_buffer;

// 		if (sorted_syn2_idx < total_number_of_synapses) {

// 			int idx_of_syn1_in_original_arrays = d_indices_of_sorted_synapses_in_orginal_arrays[sorted_syn1_idx];
// 			int idx_of_syn2_in_original_arrays = d_indices_of_sorted_synapses_in_orginal_arrays[sorted_syn2_idx];

// 			int syn1_postsyn_neuron_idx = d_postsynaptic_neuron_indices[idx_of_syn1_in_original_arrays];
// 			int syn2_postsyn_neuron_idx = d_postsynaptic_neuron_indices[idx_of_syn2_in_original_arrays];

// 			if (syn1_postsyn_neuron_idx == syn2_postsyn_neuron_idx) {

// 				// int start_idx_for_postsyn_neurons_sorted_syns = d_postsynaptic_neuron_start_indices_for_sorted_conductance_calculations[syn1_postsyn_neuron_idx];
// 				// int total_number_of_afferent_synapses_for_postsyn_neuron = d_per_neuron_afferent_synapse_count[syn1_postsyn_neuron_idx];
// 				// int end_idx_for_postsyn_neurons_sorted_syns = start_idx_for_postsyn_neurons_sorted_syns + total_number_of_afferent_synapses_for_postsyn_neuron;

// 				// if (sorted_syn2_idx < end_idx_for_postsyn_neurons_sorted_syns) {

// 					d_component_current_injections_for_each_synapse[sorted_syn1_idx] += d_component_current_injections_for_each_synapse[sorted_syn2_idx];

// 				// }

// 			}

// 		}

// 		idx += blockDim.x * gridDim.x;

// 	}
// 	__syncthreads();

// }



__global__ void move_spikes_towards_synapses_kernel(int* d_presynaptic_neuron_indices,
								int* d_delays,
								int* d_spikes_travelling_to_synapse,
								float* d_last_spike_time_of_each_neuron,
								float* d_input_neurons_last_spike_time,
								float current_time_in_seconds,
								size_t total_number_of_synapses,
								float* d_time_of_last_spike_to_reach_synapse){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_synapses) {


		int timesteps_until_spike_reaches_synapse = d_spikes_travelling_to_synapse[idx];
		timesteps_until_spike_reaches_synapse -= 1;

		if (timesteps_until_spike_reaches_synapse == 0) {
			d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds;
		}

		if (timesteps_until_spike_reaches_synapse < 0) {

			// Get presynaptic neurons last spike time
			int presynaptic_neuron_index = d_presynaptic_neuron_indices[idx];
			bool presynaptic_is_input = PRESYNAPTIC_IS_INPUT(presynaptic_neuron_index);
			float presynaptic_neurons_last_spike_time = presynaptic_is_input ? d_input_neurons_last_spike_time[CORRECTED_PRESYNAPTIC_ID(presynaptic_neuron_index, presynaptic_is_input)] : d_last_spike_time_of_each_neuron[presynaptic_neuron_index];

			if (presynaptic_neurons_last_spike_time == current_time_in_seconds){

				timesteps_until_spike_reaches_synapse = d_delays[idx];

			}
		} 

		d_spikes_travelling_to_synapse[idx] = timesteps_until_spike_reaches_synapse;

		idx += blockDim.x * gridDim.x;
	}
	__syncthreads();
}

__global__ void check_bitarray_for_presynaptic_neuron_spikes(int* d_presynaptic_neuron_indices,
								int* d_delays,
								unsigned char* d_bitarray_of_neuron_spikes,
								unsigned char* d_input_neuron_bitarray_of_neuron_spikes,
								int bitarray_length,
								int bitarray_maximum_axonal_delay_in_timesteps,
								float current_time_in_seconds,
								float timestep,
								size_t total_number_of_synapses,
								float* d_time_of_last_spike_to_reach_synapse){
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_synapses) {

		int presynaptic_neuron_index = d_presynaptic_neuron_indices[idx];
		bool presynaptic_is_input = PRESYNAPTIC_IS_INPUT(presynaptic_neuron_index);
		int delay = d_delays[idx];

		// Get offset depending upon the current timestep
		int offset_index = ((int)(round(current_time_in_seconds / timestep)) % bitarray_maximum_axonal_delay_in_timesteps) - delay;
		offset_index = (offset_index < 0) ? (offset_index + bitarray_maximum_axonal_delay_in_timesteps) : offset_index;
		int offset_byte = offset_index / 8;
		int offset_bit_pos = offset_index - (8 * offset_byte);

		// Get the correct neuron index
		int neuron_index = CORRECTED_PRESYNAPTIC_ID(presynaptic_neuron_index, presynaptic_is_input);
		
		// Check the spike
		int neuron_id_spike_store_start = neuron_index * bitarray_length;
		int check = 0;
		if (presynaptic_is_input){
			unsigned char byte = d_input_neuron_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte];
			check = ((byte >> offset_bit_pos) & 1);
			if (check == 1){
				d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds;
			}
		} else {
			unsigned char byte = d_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte];
			check = ((byte >> offset_bit_pos) & 1);
			if (check == 1){
				d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds;
			}
		}

		idx += blockDim.x * gridDim.x;
	}
	__syncthreads();
}