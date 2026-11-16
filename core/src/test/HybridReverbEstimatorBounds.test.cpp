#include <air_absorption.h>
#include <algorithm>
#include <audio_buffer.h>
#include <catch.hpp>
#include <cmath>
#include <cstdio>
#include <energy_field.h>
#include <hybrid_reverb_estimator.h>
#include <iir.h>
#include <impulse_response.h>
#include <overlap_add_convolution_effect.h>
#include <overlap_save_convolution_effect.h>
#include <propagation_medium.h>
#include <random>
#include <reconstructor.h>
#include <reverb_effect.h>
#include <triple_buffer.h>
#include <vector>



namespace {

// direct linear convolution, output length n
std::vector<double> directConv(const std::vector<float>& x, const std::vector<float>& h, size_t n)
{
    std::vector<double> y(n, 0.0);
    for (size_t i = 0; i < x.size(); ++i)
    {
        if (x[i] == 0.0f)
            continue;
        for (size_t k = 0; k < h.size() && i + k < n; ++k)
            y[i + k] += static_cast<double>(x[i]) * h[k];
    }
    return y;
}

} // namespace

TEST_CASE("IrAudit: hybrid estimator reads inside the energy field when the overlap is zero", "[IrAuditEst]")
{
    using namespace ipl;
    const auto fs = 48000;
    const auto duration = 0.5f;
    EnergyField field(duration, 1);
    std::printf("[estimator] energy field bins = %d\n", field.numBins());
    // Bin 0 of band 1 holds a marker. Reading one bin past the end of band 0 lands on it.
    field[0][1][0] = 1.0f;

    ImpulseResponse ir(duration, 1, fs);
    HybridReverbEstimator estimator(duration, fs, 512);
    Reverb reverb;
    for (auto b = 0; b < Bands::kNumBands; ++b) reverb.reverbTimes[b] = 1.0f;
    float eq[Bands::kNumBands];
    int delay = 0;
    float noAbsorption[Bands::kNumBands] = {};
    AirAbsorptionModel airAbsorption(noAbsorption, nullptr, nullptr);
    estimator.estimate(&field, reverb, airAbsorption, ir, duration, 0.0f, 1, eq, delay);
    std::printf("[estimator] eq = %g %g %g (all must be 0: the field is empty at the cutoff bin)\n", eq[0], eq[1], eq[2]);
    REQUIRE(eq[0] == 0.0f);
}


