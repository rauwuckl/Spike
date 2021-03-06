#include "ImagePoissonInputSpikingNeurons.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <algorithm> // For random shuffle

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <random> //for white noise stimuli

#include "../Helpers/FstreamWrapper.hpp"

using namespace std;


ImagePoissonInputSpikingNeurons::ImagePoissonInputSpikingNeurons() {
  filterPhases = new vector<float>();
  filterWavelengths = new vector<int>();
  filterOrientations = new vector<float>();
}

ImagePoissonInputSpikingNeurons::~ImagePoissonInputSpikingNeurons() {
  free(gabor_input_rates);
}

void ImagePoissonInputSpikingNeurons::state_update
(float current_time_in_seconds, float timestep) {
  backend()->state_update(current_time_in_seconds, timestep);
}

int ImagePoissonInputSpikingNeurons::AddGroup(neuron_parameters_struct * group_params){
  int new_group_id = PoissonInputSpikingNeurons::AddGroup(group_params);

  // Not currently used
  // image_poisson_input_spiking_neuron_parameters_struct * image_poisson_input_spiking_group_params = (image_poisson_input_spiking_neuron_parameters_struct*)group_params;
  // for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
  // }

  return new_group_id;
}


void ImagePoissonInputSpikingNeurons::AddGroupForEachGaborType(neuron_parameters_struct * group_params) {

  image_poisson_input_spiking_neuron_parameters_struct * image_poisson_input_spiking_group_params = (image_poisson_input_spiking_neuron_parameters_struct*)group_params;
  image_poisson_input_spiking_group_params->group_shape[0] = image_width;
  image_poisson_input_spiking_group_params->group_shape[1] = image_width;

  for (int gabor_index = 0; gabor_index < total_number_of_gabor_types; gabor_index++) {
    image_poisson_input_spiking_group_params->gabor_index = gabor_index;
    int new_group_id = this->AddGroup(image_poisson_input_spiking_group_params);
  }

}


void ImagePoissonInputSpikingNeurons::set_up_rates(const char * fileList, const char * filterParameters, const char * inputDirectory, float max_rate_scaling_factor) {
  #ifndef SILENCE_IMAGE_POISSON_INPUT_SPIKING_NEURONS_SETUP
  printf("--- Setting up Input Neuron Rates from Gabor files...\n");
  #endif

  // Reset (added by aki)
  total_number_of_objects = 0;
  total_number_of_transformations_per_object = 0;
  inputNames.clear();
  filterPhases->clear();
  filterWavelengths->clear();
  filterOrientations->clear();
  // Reset should also be done for gabor filtered images
  if (gabor_input_rates)
	  free(gabor_input_rates);

  load_image_names_from_file_list(fileList, inputDirectory);
  load_gabor_filter_parameters(filterParameters, inputDirectory);

  if(make_stimuli_as_random_noise){
    create_random_rates_like_in_files(max_rate_scaling_factor);
  }
  else{
    load_rates_from_files(inputDirectory, max_rate_scaling_factor);
  }
}

void ImagePoissonInputSpikingNeurons::copy_rates_to_device() {
  backend()->copy_rates_to_device();
}

void ImagePoissonInputSpikingNeurons::load_image_names_from_file_list(const char * fileList, const char * inputDirectory) {

  // Open file list
  stringstream path;
  path << inputDirectory << fileList;
  string path_string = path.str();

  ifstream fileListStream;
  fileListStream.open(path_string.c_str());

  if(fileListStream.fail()) {
    stringstream s;
    s << "Unable to open " << path_string << " for input." << endl;
    cerr << s.str();
    exit(EXIT_FAILURE);
  }

  string dirNameBase;						// The "shapeS1T2" part of "shapeS1T2.png"
  int filesLoaded = 0;
  int lastNrOfTransformsFound = 0; // For validation of file list

  // cout << "Reading file list:" << endl;

  while(getline(fileListStream, dirNameBase)) { 	// Read line from file list

    if(dirNameBase.compare("") == 0) {
      continue; // Last line may just be empty bcs of matlab script, should be break; really, but what the hell
    } else if(dirNameBase.compare("*") == 0) {
      if(lastNrOfTransformsFound != 0 && lastNrOfTransformsFound != total_number_of_transformations_per_object) {
        cerr << "Number of transforms varied in file list" << endl;
        exit(EXIT_FAILURE);
      }

      total_number_of_objects++;
      lastNrOfTransformsFound = total_number_of_transformations_per_object;
      total_number_of_transformations_per_object = 0;

      continue;
    } else {
      filesLoaded++;
      total_number_of_transformations_per_object++;
    }

    // cout << "#" << filesLoaded << " Loading: " << dirNameBase << endl;

    inputNames.push_back(dirNameBase);
  }

  total_number_of_transformations_per_object = lastNrOfTransformsFound;

  #ifndef SILENCE_IMAGE_POISSON_INPUT_SPIKING_NEURONS_SETUP
  cout << "--- --- Objects: " << total_number_of_objects << ", Transforms per Object: " << total_number_of_transformations_per_object << endl;
  #endif

  total_number_of_input_stimuli = total_number_of_objects * total_number_of_transformations_per_object;
}


void ImagePoissonInputSpikingNeurons::load_gabor_filter_parameters(const char * filterParameters, const char * inputDirectory) {


  // cout << "Reading filter parameters:" << endl;

  // Open filterParameters
  stringstream path;
  path << inputDirectory << '/' << filterParameters;
  string path_string = path.str();

  ifstream filterParametersStream;
  filterParametersStream.open(path_string.c_str());

  if(filterParametersStream.fail()) {
    stringstream s;
    s << "Unable to open " << path_string << " for input." << endl;
    cerr << s.str();
    exit(EXIT_FAILURE);
  }

  string dirNameBase;

  #ifndef SILENCE_IMAGE_POISSON_INPUT_SPIKING_NEURONS_SETUP
  cout << "--- --- Gabor Parameters:" << endl;
  #endif

  int line_index = 0;
  while(getline(filterParametersStream, dirNameBase)) {

    #ifndef SILENCE_IMAGE_POISSON_INPUT_SPIKING_NEURONS_SETUP
    cout << "--- --- --- " << dirNameBase << endl;
    #endif

    stringstream lineStream(dirNameBase);

    int num;
    while (lineStream.str().size() != 0) {

      if ((lineStream.peek() == ',') || (lineStream.peek() == '[') || (lineStream.peek() == ' ')) {
        lineStream.ignore();
      } else if (lineStream.peek() == ']') {
        break;
      } else {
        lineStream >> num;

        switch (line_index) {
        case 0:
          filterPhases->push_back((float)num);
          break;

        case 1:
          filterWavelengths->push_back(num);
          break;

        case 2:
          filterOrientations->push_back((float)num);
          break;

        case 3:
          image_width = num;
          break;
        }

      }
    }

    line_index++;

  }

  total_number_of_phases = filterPhases->size();
  total_number_of_wavelengths = filterWavelengths->size();
  total_number_of_orientations = filterOrientations->size();
  total_number_of_gabor_types = total_number_of_phases*total_number_of_wavelengths*total_number_of_orientations;

  total_number_of_rates_per_image = total_number_of_gabor_types * image_width * image_width;
  total_number_of_rates = total_number_of_input_stimuli * total_number_of_rates_per_image;

  // printf("\ntotal_number_of_rates: %d\n", total_number_of_rates);
}


void ImagePoissonInputSpikingNeurons::load_rates_from_files(const char * inputDirectory, float max_rate_scaling_factor) {


  gabor_input_rates = (float *)malloc(total_number_of_rates*sizeof(float));
  int zero_count = 0;

  for(int image_index = 0; image_index < total_number_of_input_stimuli; image_index++) {

    float total_activation_for_image = 0.0;

    int image_starting_index = image_index * total_number_of_rates_per_image;
    // printf("image_starting_index: %d\n", image_starting_index);

    // cout << "Loading Rates for Image #" << image_index << endl;

    for(int orientation_index = 0; orientation_index < total_number_of_orientations; orientation_index++) {

      for(int wavelength_index = 0; wavelength_index < total_number_of_wavelengths; wavelength_index++) {

        for(int phase_index = 0; phase_index < total_number_of_phases; phase_index++) {

          int gabor_index = calculate_gabor_index(orientation_index,wavelength_index,phase_index);
          int start_index_for_current_gabor_image = image_starting_index + gabor_index * image_width * image_width;

          // printf("ORIENTATION: %d\n", orientation_index);
          // printf("WAVELENGTH: %d\n", wavelength_index);
          // printf("PHASE: %d\n\n", phase_index);
          // printf("GABOR_INDEX: %d\n", gabor_index);

          // Read input to network
          ostringstream dirStream;

          dirStream << inputDirectory << "Filtered/" << inputNames[image_index] << ".flt" << "/"
                    << inputNames[image_index] << '.' << filterWavelengths->at(wavelength_index) << '.'
                    << filterOrientations->at(orientation_index) << '.' << filterPhases->at(phase_index) << ".gbo";

          string t = dirStream.str();

          // Open&Read gabor filter file
          fstreamWrapper gaborStream;

          try {

            gaborStream.open(t.c_str(), std::ios_base::in | std::ios_base::binary);

            for(int image_x = 0; image_x < image_width; image_x++)
              for(int image_y = 0; image_y < image_width; image_y++) {

                float rate;
                gaborStream >> rate;

                // printf("rate: %f\n", rate);
                if (rate < 0.000001) zero_count++;

                if(rate < 0) {
                  cerr << "Negative firing loaded from filter!!!" << endl;
                  exit(EXIT_FAILURE);
                }

                int element_index = start_index_for_current_gabor_image + image_x + image_y * image_width;

                total_activation_for_image += rate;

                // Rates from Matlab lie between 0 and 1, so multiply by max number of spikes per second in cortex
                gabor_input_rates[element_index] = rate * max_rate_scaling_factor;
              }

          } catch (fstream::failure e) {
            stringstream s;
            s << "Unable to open/read from " << t << " for gabor input: " << e.what();
            cerr << s.str();
            exit(EXIT_FAILURE);
          }
        }
      }
    }
    // printf("total_activation_for_image: %f\n", total_activation_for_image);
  }
}

void ImagePoissonInputSpikingNeurons::create_random_rates_like_in_files(float max_rate_scaling_factor) {
  /* Behaves from outside same as load_rates_from_files. But instead of actually reading the rates it just makes up random rates uniformly distributed
  *
  */
  std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!! STIMULI ARE REPLACED BY RANDOM NOISE\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

  //Set up random number generation
  std::random_device rand_device;
  // std::cout << "random device says: "<< rand_device() << std::endl;
  std::mt19937 e2(rand_device());
  std::exponential_distribution<> dist(1);
  std::bernoulli_distribution sparsity_dist(0.067);




  gabor_input_rates = (float *)malloc(total_number_of_rates*sizeof(float));

  for(int image_index = 0; image_index < total_number_of_input_stimuli; image_index++) {

    float total_activation_for_image = 0.0;

    int image_starting_index = image_index * total_number_of_rates_per_image;
    // printf("image_starting_index: %d\n", image_starting_index);

    // cout << "Loading Rates for Image #" << image_index << endl;

    for(int orientation_index = 0; orientation_index < total_number_of_orientations; orientation_index++) {

      for(int wavelength_index = 0; wavelength_index < total_number_of_wavelengths; wavelength_index++) {

        for(int phase_index = 0; phase_index < total_number_of_phases; phase_index++) {
          int nonzero_count = 0;

          int gabor_index = calculate_gabor_index(orientation_index,wavelength_index,phase_index);
          int start_index_for_current_gabor_image = image_starting_index + gabor_index * image_width * image_width;

          for(int image_x = 0; image_x < image_width; image_x++){
            for(int image_y = 0; image_y < image_width; image_y++) {

              float rate;
              //RANDOMLY invent the rate instead of actually reading it
              if( sparsity_dist(e2)){
                rate = dist(e2);
              }else{
                rate = 0.0;
              }

              // printf("rate: %f\n", rate);
              if (rate > 0.000001) nonzero_count++;

              if(rate < 0) {
                cerr << "Negative firing loaded from filter!!!" << endl;
                exit(EXIT_FAILURE);
              }

              int element_index = start_index_for_current_gabor_image + image_x + image_y * image_width;

              total_activation_for_image += rate;

              // Rates from Matlab lie between 0 and 1, so multiply by max number of spikes per second in cortex
              gabor_input_rates[element_index] = rate * max_rate_scaling_factor;
            }
          }
          // std::cout << " nonzero count " << nonzero_count << std::endl;
        }
      }
    }
    // printf("total_activation_for_image: %f\n", total_activation_for_image);
  }
  // printf("--- --- Proportion of input rates 0.0: %f\n", (float)zero_count/(float)total_number_of_rates);
}

int ImagePoissonInputSpikingNeurons::calculate_gabor_index(int orientationIndex, int wavelengthIndex, int phaseIndex) {
  return orientationIndex * (total_number_of_wavelengths * total_number_of_phases) + wavelengthIndex * total_number_of_phases + phaseIndex;
}


SPIKE_MAKE_INIT_BACKEND(ImagePoissonInputSpikingNeurons);
