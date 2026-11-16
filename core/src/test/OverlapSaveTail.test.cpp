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

TEST_CASE("IrAudit: tail of the overlap-save convolution", "[IrAudit]")
{
    using namespace ipl;
    const auto fs = 48000;
    const auto frameSize = 512;
    const auto irSize = 4 * frameSize;

    ImpulseResponse ir(static_cast<float>(irSize) / fs, 0, fs);
    std::mt19937 rng(2);
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

    // Reference: keep feeding silence through apply().
    // Test: one frame with a unit impulse in the middle, then tail().
    const auto impulsePos = frameSize / 2;
    AudioBuffer in(1, frameSize);
    AudioBuffer out(1, frameSize);

    OverlapSaveConvolutionEffectParams params{};
    params.fftIR = &fir;
    params.numChannels = 1;
    params.numSamples = irSize;

    std::vector<float> ref;
    {
        OverlapSaveConvolutionEffect effect(AudioSettings{fs, frameSize}, OverlapSaveConvolutionEffectSettings{1, irSize});
        memset(in[0], 0, frameSize * sizeof(float));
        effect.apply(params, in, out);
        for (auto f = 0; f < 8; ++f)
        {
            memset(in[0], 0, frameSize * sizeof(float));
            if (f == 0)
                in[0][impulsePos] = 1.0f;
            effect.apply(params, in, out);
            ref.insert(ref.end(), out[0], out[0] + frameSize);
        }
    }

    std::vector<float> tl;
    {
        partitioner.partition(ir, 1, irSize, *fir.writeBuffer);
        fir.commitWriteBuffer();
        OverlapSaveConvolutionEffect effect(AudioSettings{fs, frameSize}, OverlapSaveConvolutionEffectSettings{1, irSize});
        memset(in[0], 0, frameSize * sizeof(float));
        effect.apply(params, in, out);
        in[0][impulsePos] = 1.0f;
        effect.apply(params, in, out);
        tl.insert(tl.end(), out[0], out[0] + frameSize);
        auto state = AudioEffectState::TailRemaining;
        auto n = 0;
        while (state == AudioEffectState::TailRemaining && n < 20)
        {
            state = effect.tail(out);
            tl.insert(tl.end(), out[0], out[0] + frameSize);
            ++n;
        }
        std::printf("[tail] tail frames produced = %d (the response spans %d frames after the first)\n", n,
                    (impulsePos + irSize - 1) / frameSize);
    }

    // sanity: apply() path equals h shifted by the impulse position
    double errRef = 0.0;
    for (auto i = 0; i < irSize; ++i)
        errRef = std::max(errRef, static_cast<double>(std::fabs(ref[impulsePos + i] - h[i])));
    REQUIRE(errRef < 1e-3);

    double errTail = 0.0, eRef = 0.0, eErr = 0.0;
    for (size_t i = 0; i < tl.size() && i < ref.size(); ++i)
    {
        errTail = std::max(errTail, static_cast<double>(std::fabs(tl[i] - ref[i])));
        eRef += static_cast<double>(ref[i]) * ref[i];
        eErr += static_cast<double>(tl[i] - ref[i]) * (tl[i] - ref[i]);
    }
    for (size_t f = 0; f * frameSize < tl.size() && f < 6; ++f)
    {
        double er = 0, et = 0, ee = 0;
        for (auto i = 0; i < frameSize; ++i)
        {
            auto a = ref[f * frameSize + i], b = tl[f * frameSize + i];
            er += a * a; et += b * b; ee += (a - b) * (a - b);
        }
        std::printf("[tail] frame %zu: ref energy %g, tail energy %g, error energy %g\n", f, er, et, ee);
    }
    std::printf("[tail] max |tail - reference| = %g, relative error energy = %g\n", errTail, eErr / eRef);
    REQUIRE(eErr / eRef < 1e-6);
}

