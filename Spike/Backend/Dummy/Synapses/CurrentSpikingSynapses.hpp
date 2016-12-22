#pragma once

#include "Spike/Synapses/CurrentSpikingSynapses.hpp"
#include "SpikingSynapses.hpp"

namespace Backend {
  namespace Dummy {
    class CurrentSpikingSynapses : public virtual ::Backend::Dummy::SpikingSynapses,
                                   public virtual ::Backend::CurrentSpikingSynapses {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(CurrentSpikingSynapses);

      void prepare() override {
        SpikingSynapses::prepare();
      }

      void calculate_postsynaptic_current_injection(::SpikingNeurons * neurons, float current_time_in_seconds, float timestep) final {
      }

      void reset_state() override {
        SpikingSynapses::reset_state();
      }

      void push_data_front() override {
        SpikingSynapses::push_data_front();
      }

      void pull_data_back() override {
        SpikingSynapses::pull_data_back();
      }
    };
  } // namespace Dummy
} // namespace Backend

