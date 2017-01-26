#include "Optimiser.hpp"

#include "../SpikeAnalyser/SpikeAnalyser.hpp"
#include "../Helpers/TimerWithMessages.hpp"
#include "../Helpers/TerminalHelpers.hpp"
#include "../Helpers/Memory.hpp"
#include <sys/stat.h>
#include <sstream>
#include <iomanip>



// Constructors
Optimiser::Optimiser(SpikingModel* spiking_model_parameter, string full_output_directory_parameter) {

	spiking_model = spiking_model_parameter;
	full_output_directory = full_output_directory_parameter;

}


void Optimiser::AddOptimisationStage(Optimiser_Options * optimisation_stage_options, Simulator_Options * simulator_options_parameter) {

	int new_optimisation_stage = number_of_optimisation_stages;

	number_of_optimisation_stages++;

	simulator_options_for_each_optimisation_stage = (Simulator_Options**)realloc(simulator_options_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(Simulator_Options*));
	model_pointers_to_be_optimised_for_each_optimisation_stage = (float**)realloc(model_pointers_to_be_optimised_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(float*));
	synapse_bool_pointers_to_turn_on_for_each_optimisation_stage = (bool**)realloc(synapse_bool_pointers_to_turn_on_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(bool*));
	number_of_non_input_layers_to_simulate_for_each_optimisation_stage = (int*)realloc(number_of_non_input_layers_to_simulate_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(int));
	index_of_neuron_group_of_interest_for_each_optimisation_stage = (int*)realloc(index_of_neuron_group_of_interest_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(int));
	initial_optimisation_parameter_min_for_each_optimisation_stage = (float*)realloc(initial_optimisation_parameter_min_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(float));
	initial_optimisation_parameter_max_for_each_optimisation_stage = (float*)realloc(initial_optimisation_parameter_max_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(float));
	ideal_output_scores_for_each_optimisation_stage = (float*)realloc(ideal_output_scores_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(float));
	optimisation_minimum_error_for_each_optimisation_stage = (float*)realloc(optimisation_minimum_error_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(float));
	positive_effect_of_postive_change_in_parameter_for_each_optimisation_stage = (bool*)realloc(positive_effect_of_postive_change_in_parameter_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(bool));
	score_to_use_for_each_optimisation_stage = (int*)realloc(score_to_use_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(int));

	simulator_options_for_each_optimisation_stage[new_optimisation_stage] = simulator_options_parameter;
	model_pointers_to_be_optimised_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->model_pointer_to_be_optimised;
	synapse_bool_pointers_to_turn_on_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->synapse_bool_pointer_to_turn_on;
	number_of_non_input_layers_to_simulate_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->number_of_non_input_layers_to_simulate;
	index_of_neuron_group_of_interest_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->index_of_neuron_group_of_interest;
	initial_optimisation_parameter_min_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->initial_optimisation_parameter_min;
	initial_optimisation_parameter_max_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->initial_optimisation_parameter_max;
	ideal_output_scores_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->ideal_output_score;
	optimisation_minimum_error_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->optimisation_minimum_error;
	positive_effect_of_postive_change_in_parameter_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->positive_effect_of_postive_change_in_parameter;
	score_to_use_for_each_optimisation_stage[new_optimisation_stage] = (int)optimisation_stage_options->score_to_use;

	final_optimal_parameter_for_each_optimisation_stage = (float*)realloc(final_optimal_parameter_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(float));
	final_iteration_count_for_each_optimisation_stage = (int*)realloc(final_iteration_count_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(int));

}


void Optimiser::RunOptimisation(int start_optimisation_stage_index, bool test_last_spikes_match) {

	string previous_test_parameter_values_for_stage_FILE_NAME = full_output_directory + "PREVIOUS_TEST_PARAMETER_VALUES_FOR_STAGE";
	ofstream previous_test_parameter_values_for_stage_file_STREAM;
	previous_test_parameter_values_for_stage_file_STREAM.open((previous_test_parameter_values_for_stage_FILE_NAME + ".txt"), ios::out | ios::binary | ios::trunc);

	string previous_firing_rates_for_stage_FILE_NAME = full_output_directory + "PREVIOUS_FIRING_RATES_FOR_STAGE";
	ofstream previous_firing_rates_for_stage_STREAM;
	previous_firing_rates_for_stage_STREAM.open((previous_firing_rates_for_stage_FILE_NAME + ".txt"), ios::out | ios::binary | ios::trunc);


	for (int optimisation_stage = start_optimisation_stage_index; optimisation_stage < number_of_optimisation_stages; optimisation_stage++) {

		float optimisation_parameter_min = initial_optimisation_parameter_min_for_each_optimisation_stage[optimisation_stage];
		float optimisation_parameter_max = initial_optimisation_parameter_max_for_each_optimisation_stage[optimisation_stage];
		float optimisation_ideal_output_score = ideal_output_scores_for_each_optimisation_stage[optimisation_stage];

		int iteration_count_for_optimisation_stage = -1;;

		while (true) {

			// printf("Backend::total_memory(): %lu\n", Backend::total_memory());

			iteration_count_for_optimisation_stage++;

			print_line_of_dashes_with_blank_lines_either_side();
			
			setup_optimisation_stage_specific_model_parameters(optimisation_stage);
		
			float test_optimisation_parameter_value = (optimisation_parameter_max + optimisation_parameter_min) / 2.0;
			*model_pointers_to_be_optimised_for_each_optimisation_stage[optimisation_stage] = test_optimisation_parameter_value;
			if (test_last_spikes_match) *model_pointers_to_be_optimised_for_each_optimisation_stage[optimisation_stage] = 0.1;

			printf("OPTIMISATION ITERATION BEGINNING... \nOptimisation Stage: %d\nIteration Count for Optimisation Stage: %d\nNew Test Optimisaton Parameter: %f\n", optimisation_stage, iteration_count_for_optimisation_stage, test_optimisation_parameter_value);

			print_line_of_dashes_with_blank_lines_either_side();

			// FINALISE MODEL + COPY TO DEVICE
			spiking_model->finalise_model();

			
			simulator_options_for_each_optimisation_stage[optimisation_stage]->run_simulation_general_options->delete_spike_analyser_on_simulator_destruction = !test_last_spikes_match;


			// CREATE SIMULATOR
			Simulator * simulator = new Simulator(spiking_model, simulator_options_for_each_optimisation_stage[optimisation_stage]);

			// RUN SIMULATION
			simulator->RunSimulation();

			// CALCULATE AVERAGES + OPTIMISATION OUTPUT SCORE
			SpikeAnalyser *new_spike_analyser = simulator->spike_analyser;
			new_spike_analyser->calculate_various_neuron_spike_totals_and_averages(simulator_options_for_each_optimisation_stage[optimisation_stage]->run_simulation_general_options->presentation_time_per_stimulus_per_epoch);

			

			// BONES FOR TESTING IF SPIKES MATCH BETWEEN ITERATIONS
			// if (test_last_spikes_match) {

			// 	printf("TEST: Comparte spike totals to last optimisation iteration\n");

			// 	if (iteration_count_for_optimisation_stage > 0) {
			// 		int total_wrong_count = 0;
			// 		for (int stimulus_index = 0; stimulus_index < four_layer_vision_spiking_model->input_spiking_neurons->total_number_of_input_stimuli; stimulus_index++) {
			// 			for (int neuron_index = 0; neuron_index < 4096; neuron_index++) {
			// 				if (new_spike_analyser->per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index] != spike_analyser_from_last_optimisation_stage->per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index]) {
			// 				// if (spike_analyser_from_last_optimisation_stage->per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index]) {
			// 					printf("new_spike_analyser->per_stimulus_per_neuron_spike_counts[%d][%d]: %d\n", stimulus_index, neuron_index, new_spike_analyser->per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index]);
			// 					printf("spike_analyser_from_last_optimisation_stage->per_stimulus_per_neuron_spike_counts[%d][%d]: %d\n", stimulus_index, neuron_index, spike_analyser_from_last_optimisation_stage->per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index]);
			// 					total_wrong_count++;
			// 					printf("neuron_index: %d\n", neuron_index);
			// 					print_message_and_exit("Test failure: Spike totals do not match");
			// 				}		
			// 			}
			// 		}
			// 		printf("total_wrong_count: %d\n", total_wrong_count);
			// // 	}

			// 	delete spike_analyser_from_last_optimisation_stage;
			// 	spike_analyser_from_last_optimisation_stage = new_spike_analyser;

			// }

			


			// SET APPROPRIATE OPTIMISATION OUTPUT SCORE
			float optimisation_output_score = 0.0;
			int index_of_neuron_group_of_interest = index_of_neuron_group_of_interest_for_each_optimisation_stage[optimisation_stage];

			switch(score_to_use_for_each_optimisation_stage[optimisation_stage]) {

				case SCORE_TO_USE_average_number_of_spikes_per_neuron_group_per_second:
					optimisation_output_score = new_spike_analyser->average_number_of_spikes_per_neuron_group_per_second[index_of_neuron_group_of_interest];
					break;

				case SCORE_TO_USE_max_number_of_spikes_per_neuron_group_per_second:
					optimisation_output_score = new_spike_analyser->max_number_of_spikes_per_neuron_group_per_second[index_of_neuron_group_of_interest];
					break;

				case SCORE_TO_USE_average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons:
					optimisation_output_score = new_spike_analyser->average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons[index_of_neuron_group_of_interest];
					break;

				case SCORE_TO_USE_running_count_of_non_silent_neurons_per_neuron_group:
					// optimisation_output_score = new_spike_analyser->running_count_of_non_silent_neurons_per_neuron_group[index_of_neuron_group_of_interest];
					optimisation_output_score = (float)(new_spike_analyser->total_number_of_spikes_per_neuron_group[0])/356.0;

			}

			printf("OPTIMISATION ITERATION COMPLETED...\nTest Optimisation Parameter Value: %.16f\nOptimisation Output Score: %f\nOptimisation Ideal Output Score: %f\n", test_optimisation_parameter_value, optimisation_output_score, optimisation_ideal_output_score);
			

			// WRITE OPTIMISATION STATE TO FILE
			string optimisation_state_FILE_NAME = full_output_directory + "OPTIMISATION_STATE";
			ofstream optimisation_state_STREAM;
			optimisation_state_STREAM.open((optimisation_state_FILE_NAME + ".txt"), ios::out | ios::binary | ios::trunc);

			optimisation_state_STREAM << "optimisation_stage: " << to_string(optimisation_stage) << endl 
										<< "iteration_count_for_optimisation_stage: " << to_string(iteration_count_for_optimisation_stage) << endl << endl 
										<< "test_optimisation_parameter_value: " << to_string(test_optimisation_parameter_value) << endl << endl
										<< "optimisation_output_score: " << to_string(optimisation_output_score) << endl
										<< "optimisation_ideal_output_score: " << to_string(optimisation_ideal_output_score) << endl
										<< "optimisation_minimum_error_for_each_optimisation_stage[optimisation_stage]: " << to_string(optimisation_minimum_error_for_each_optimisation_stage[optimisation_stage]) << endl << endl;
			optimisation_state_STREAM.close();


			// WRITE TEST PARAMETER VALUES TO FILE
			previous_test_parameter_values_for_stage_file_STREAM << to_string(test_optimisation_parameter_value) << ", " << to_string(optimisation_output_score) << ", " << to_string(optimisation_ideal_output_score) << endl;
			

			// WRITE FIRING RATES FOR ALL LAYERS TO FILE
			for (int neuron_group_index = 0; neuron_group_index < spiking_model->spiking_neurons->total_number_of_groups; neuron_group_index++) {
				previous_firing_rates_for_stage_STREAM << to_string(new_spike_analyser->average_number_of_spikes_per_neuron_group_per_second[neuron_group_index]) << ", ";
			}
			previous_firing_rates_for_stage_STREAM << endl;



			// CHECK IF OPTIMISATION STAGE COMPLETE
			float difference_between_ideal_score_and_output_score = optimisation_ideal_output_score - optimisation_output_score; // Supposing the function we are trying to optimise is monotonic, the sign of this value gives the direction that the optimisation must move in.
			if (fabs(difference_between_ideal_score_and_output_score) < optimisation_minimum_error_for_each_optimisation_stage[optimisation_stage]) {

				//OPTIMISATION STAGE COMPLETE
			
				// STORE FINAL OPTIMAL PARAMETER
				final_optimal_parameter_for_each_optimisation_stage[optimisation_stage] = test_optimisation_parameter_value;

				// WRITE OPTIMISTION STAGE PARAMETERS TO FILE
				write_optimisation_stage_parameters_to_file(optimisation_stage);
				break;
			
			}

			// UPDATE RANGE ACCORDINGLY IF OPTIMISATION NOT COMPLETE
			float effect_direction_factor = positive_effect_of_postive_change_in_parameter_for_each_optimisation_stage[optimisation_stage] ? 1.0 : -1.0;
			if (effect_direction_factor * difference_between_ideal_score_and_output_score > 0) {
			
				optimisation_parameter_min = test_optimisation_parameter_value;
			
			} else {
			
				optimisation_parameter_max = test_optimisation_parameter_value;
			
			}

					
			delete simulator;

			print_line_of_dashes_with_blank_lines_either_side();
			printf("Backend::memory_free_bytes(): %lu", Backend::memory_free_bytes());
			print_line_of_dashes_with_blank_lines_either_side();

		}

		previous_test_parameter_values_for_stage_file_STREAM.close();
		previous_firing_rates_for_stage_STREAM.close();

		if (mkdir((full_output_directory + "/PREVIOUS_TEST_PARAMETER_VALUES_FOR_STAGES/").c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP | S_IXGRP | S_IROTH | S_IWOTH | S_IXOTH)==0)
			printf("\nDirectory created\n");
		FILE * infile  = fopen((previous_test_parameter_values_for_stage_FILE_NAME + ".txt").c_str(),  "rb");
		string previous_test_parameter_values_for_stage_FILE_NAME = full_output_directory + "/PREVIOUS_TEST_PARAMETER_VALUES_FOR_STAGES/" + to_string(optimisation_stage) + ".txt";
		FILE * outfile = fopen(previous_test_parameter_values_for_stage_FILE_NAME.c_str(), "wb");
		file_copy_and_close(outfile, infile);

    	if (mkdir((full_output_directory + "/PREVIOUS_FIRING_RATES_FOR_STAGES/").c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP | S_IXGRP | S_IROTH | S_IWOTH | S_IXOTH)==0)
			printf("\nDirectory created\n");
		FILE * infile3  = fopen((previous_firing_rates_for_stage_FILE_NAME + ".txt").c_str(),  "rb");
		string previous_firing_rates_for_stage_FILE_NAME = full_output_directory + "/PREVIOUS_FIRING_RATES_FOR_STAGES/" + to_string(optimisation_stage) + ".txt";
		FILE * outfile3 = fopen(previous_firing_rates_for_stage_FILE_NAME.c_str(), "wb");
		file_copy_and_close(outfile3, infile3);


		final_iteration_count_for_each_optimisation_stage[optimisation_stage] = iteration_count_for_optimisation_stage; 

		print_line_of_dashes_with_blank_lines_either_side();
		
		printf("FINAL OPTIMAL PARAMETER FOR OPTIMISATION STAGE %d: %.12f\n", optimisation_stage, final_optimal_parameter_for_each_optimisation_stage[optimisation_stage]);
		printf("TOTAL OPTIMISATION ITERATIONS FOR OPTIMISATION STAGE %d: %d\n", optimisation_stage, final_iteration_count_for_each_optimisation_stage[optimisation_stage]);
		printf("Backend::memory_free_bytes(): %lu\n", Backend::memory_free_bytes());

	}

	printf("\n");
	for (int optimisation_stage = 0; optimisation_stage < number_of_optimisation_stages; optimisation_stage++) {

		printf("FINAL OPTIMAL PARAMETER FOR OPTIMISATION STAGE %d: %.12f\n", optimisation_stage, final_optimal_parameter_for_each_optimisation_stage[optimisation_stage]);
		printf("TOTAL OPTIMISATION ITERATIONS FOR OPTIMISATION STAGE %d: %d\n", optimisation_stage, final_iteration_count_for_each_optimisation_stage[optimisation_stage]);

	}

	print_line_of_dashes_with_blank_lines_either_side();

}


void Optimiser::write_final_optimisation_parameters_to_file(string full_output_directory) {

	for (int optimisation_stage = 0; optimisation_stage < number_of_optimisation_stages; optimisation_stage++) {

		write_optimisation_stage_parameters_to_file(optimisation_stage);

	}

}


void Optimiser::write_optimisation_stage_parameters_to_file(int optimisation_stage) {

	string optimisation_state_FILE_NAME = full_output_directory + to_string(optimisation_stage);

	ofstream spikeidfile;
	spikeidfile.open((optimisation_state_FILE_NAME + ".txt"), ios::out | ios::binary);

	stringstream stream;
	stream << fixed << setprecision(12) << final_optimal_parameter_for_each_optimisation_stage[optimisation_stage];
	string s = stream.str();

	spikeidfile << s << endl;
	// spikeidfile << to_string(final_optimal_parameter_for_each_optimisation_stage[optimisation_stage]) << endl;

	spikeidfile.close();

}


void Optimiser::setup_optimisation_stage_specific_model_parameters(int optimisation_stage) {

	*synapse_bool_pointers_to_turn_on_for_each_optimisation_stage[optimisation_stage] = true;
	*model_pointers_to_be_optimised_for_each_optimisation_stage[optimisation_stage] = final_optimal_parameter_for_each_optimisation_stage[optimisation_stage];

	spiking_model->setup_optimisation_stage_specific_model_parameters(optimisation_stage, this);



}

void Optimiser::file_copy_and_close(FILE *dest, FILE *src)
{
	char ch;
	rewind(src);
	rewind(dest);
	while((ch=fgetc(src)) != EOF)
	{
		fputc(ch, dest);
	}
	fflush(dest);
	rewind(src);
	rewind(dest);

	fclose(dest);
	fclose(src);
}


// void Optimiser::set_final_optimised_parameters_network() {

// 	for (int optimisation_stage_index = 0; optimisation_stage_index < number_of_optimisation_stages; optimisation_stage_index++) {
		
// 		setup_optimisation_stage_specific_model_parameters(optimisation_stage_index);

// 	}

// }
