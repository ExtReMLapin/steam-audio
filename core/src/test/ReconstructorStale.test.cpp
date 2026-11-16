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

TEST_CASE("IrAudit: Gaussian reconstruction leaves stale data in bins under the threshold", "[IrAudit]")
{
    using namespace ipl;
    const auto fs = 48000;
    const auto duration = 0.5f;

    EnergyField withEnergy(duration, 1);
    EnergyField sparse(duration, 1);
    for (auto b = 0; b < Bands::kNumBands; ++b)
        for (auto bin = 0; bin < withEnergy.numBins(); ++bin)
        {
            withEnergy[0][b][bin] = 1.0f;
            sparse[0][b][bin] = (bin < 5) ? 1.0f : 0.0f; // energy stops after 50 ms
        }

    Reconstructor reconstructor(duration, 1, fs);
    const float zeroCoeffs[3] = {0.0f, 0.0f, 0.0f};
    AirAbsorptionModel absorption(zeroCoeffs, nullptr, nullptr);
    ImpulseResponse ir(duration, 1, fs);

    const EnergyField* f1[] = {&withEnergy};
    ImpulseResponse* irs[] = {&ir};
    const float* curves[] = {nullptr};
    reconstructor.reconstruct(1, f1, curves, &absorption, irs, ReconstructionType::Gaussian, duration, 1);

    const EnergyField* f2[] = {&sparse};
    reconstructor.reconstruct(1, f2, curves, &absorption, irs, ReconstructionType::Gaussian, duration, 1);

    // After 100 ms the second field has no energy at all: the response must be silent there.
    double e = 0.0;
    for (auto i = static_cast<int>(0.1f * fs); i < ir.numSamples(); ++i)
        e += static_cast<double>(ir[0][i]) * ir[0][i];
    std::printf("[gaussian-stale] energy after 100 ms in a response whose field is empty after 50 ms = %g\n", e);
    REQUIRE(e < 1e-6);
}




TEST_CASE("IrAudit: linear reconstruction from a field shorter than the response leaves stale data", "[IrAudit]")
{
    using namespace ipl;
    const auto fs = 48000;
    const auto duration = 0.5f;

    EnergyField longField(duration, 0);
    EnergyField shortField(0.2f, 0);
    for (auto b = 0; b < Bands::kNumBands; ++b)
    {
        for (auto bin = 0; bin < longField.numBins(); ++bin)
            longField[0][b][bin] = 1.0f;
        for (auto bin = 0; bin < shortField.numBins(); ++bin)
            shortField[0][b][bin] = 1.0f;
    }

    Reconstructor reconstructor(duration, 0, fs);
    const float zeroCoeffs[3] = {0.0f, 0.0f, 0.0f};
    AirAbsorptionModel absorption(zeroCoeffs, nullptr, nullptr);
    ImpulseResponse ir(duration, 0, fs);
    const float* curves[] = {nullptr};
    ImpulseResponse* irs[] = {&ir};

    const EnergyField* f1[] = {&longField};
    reconstructor.reconstruct(1, f1, curves, &absorption, irs, ReconstructionType::Linear, duration, 0);
    {
        double e1 = 0.0;
        for (auto i = static_cast<int>(0.25f * fs); i < ir.numSamples(); ++i)
            e1 += static_cast<double>(ir[0][i]) * ir[0][i];
        std::printf("[linear-stale] after the first reconstruction: energy after 250 ms = %g (bins %d vs %d)\n", e1, longField.numBins(), shortField.numBins());
    }
    const EnergyField* f2[] = {&shortField};
    reconstructor.reconstruct(1, f2, curves, &absorption, irs, ReconstructionType::Linear, duration, 0);

    // Past the end of the second field (plus the tail of the band filters) the response must be silent.
    double e = 0.0;
    for (auto i = static_cast<int>(0.25f * fs); i < ir.numSamples(); ++i)
        e += static_cast<double>(ir[0][i]) * ir[0][i];
    std::printf("[linear-stale] energy after 250 ms in a response whose field ends at 200 ms = %g\n", e);
    REQUIRE(e < 1e-6);
}
