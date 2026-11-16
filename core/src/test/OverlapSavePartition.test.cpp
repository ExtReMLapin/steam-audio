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

TEST_CASE("IrAudit: partitioner with an IR size that is not a multiple of the frame size", "[IrAudit]")
{
    using namespace ipl;
    const auto fs = 48000;
    const auto frameSize = 512;
    const auto irSize = 24000; // 46 blocks + 448 samples (what PBA uses: 0.5 s at 48 kHz, 512 frames)

    ImpulseResponse ir(static_cast<float>(irSize) / fs, 0, fs);
    REQUIRE(ir.numSamples() >= irSize);

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> h(irSize);
    for (auto i = 0; i < irSize; ++i)
    {
        h[i] = dist(rng);
        ir[0][i] = h[i];
    }

    TripleBuffer<OverlapSaveFIR> fir;
    fir.initBuffers(1, irSize, frameSize);

    OverlapSavePartitioner partitioner(frameSize);
    partitioner.partition(ir, 1, irSize, *fir.writeBuffer);
    fir.commitWriteBuffer();

    OverlapSaveConvolutionEffect effect(AudioSettings{fs, frameSize}, OverlapSaveConvolutionEffectSettings{1, irSize});

    const auto numFrames = OverlapSaveConvolutionEffect::numBlocks(frameSize, irSize) + 2;
    AudioBuffer in(1, frameSize);
    AudioBuffer out(1, frameSize);

    std::vector<float> x(static_cast<size_t>(numFrames) * frameSize, 0.0f);
    x[0] = 1.0f; // unit impulse
    std::vector<float> y;

    OverlapSaveConvolutionEffectParams params{};
    params.fftIR = &fir;
    params.numChannels = 1;
    params.numSamples = irSize;

    // the first frame after a new IR is cross-faded from silence: let it pass
    memset(in[0], 0, frameSize * sizeof(float));
    effect.apply(params, in, out);

    for (auto f = 0; f < numFrames; ++f)
    {
        memcpy(in[0], &x[static_cast<size_t>(f) * frameSize], frameSize * sizeof(float));
        effect.apply(params, in, out);
        y.insert(y.end(), out[0], out[0] + frameSize);
    }

    // error inside the IR
    double maxErrInside = 0.0;
    for (auto i = 0; i < irSize; ++i)
        maxErrInside = std::max(maxErrInside, static_cast<double>(std::fabs(y[i] - h[i])));
    // energy after the end of the IR (must be exactly 0 for a unit impulse)
    double tailEnergy = 0.0;
    for (size_t i = irSize; i < y.size(); ++i)
        tailEnergy += static_cast<double>(y[i]) * y[i];

    std::printf("[partition] max error inside IR = %g, energy past the end of the IR = %g\n", maxErrInside, tailEnergy);
    REQUIRE(maxErrInside < 1e-3);
    REQUIRE(tailEnergy < 1e-6);
}

