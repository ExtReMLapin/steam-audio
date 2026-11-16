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

TEST_CASE("IrAudit: linear reconstruction keeps the energy of a bin inside the bin", "[IrAuditEnv]")
{
    using namespace ipl;
    const auto fs = 48000;
    const auto duration = 0.5f;
    const auto order = 0;
    const auto k = 20;

    EnergyField field(duration, order);
    for (auto b = 0; b < Bands::kNumBands; ++b)
        field[0][b][k] = 0.01f; // an isolated burst of energy in one bin

    const float zeroCoeffs[3] = {0.0f, 0.0f, 0.0f};
    AirAbsorptionModel none(zeroCoeffs, nullptr, nullptr);
    Reconstructor rec(duration, order, fs);
    ImpulseResponse ir(duration, order, fs);
    const EnergyField* f[] = {&field};
    const float* curves[] = {nullptr};
    ImpulseResponse* a[] = {&ir};
    rec.reconstruct(1, f, curves, &none, a, ReconstructionType::Linear, duration, order);

    const auto perBin = static_cast<int>(std::ceil(0.01f * fs));
    double e[4];
    double total = 0;
    for (auto j = 0; j < 4; ++j)
    {
        e[j] = 0;
        for (auto i = (k - 1 + j) * perBin; i < (k + j) * perBin; ++i)
            e[j] += static_cast<double>(ir[0][i]) * ir[0][i];
        total += e[j];
    }
    std::printf("[envelope] energy in bins %d..%d: %g %g %g %g (bin %d holds %.0f %% of the energy)\n", k - 1, k + 2, e[0], e[1], e[2], e[3], k, 100.0 * e[1] / total);
    REQUIRE(e[1] / total > 0.8);
}
