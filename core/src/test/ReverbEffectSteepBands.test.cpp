//
// Copyright 2017-2023 Valve Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <catch.hpp>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <utility>
#include <vector>

#include <audio_buffer.h>
#include <hybrid_reverb_effect.h>
#include <reverb_effect.h>

#if !defined(IPL_ENABLE_OCTAVE_BANDS)

namespace {

const auto kSamplingRate = 48000;
const auto kFrameSize = 256;

// Where each band is measured: away from the band edges (800 Hz and 8 kHz).
const double kMeasuredRanges[ipl::Bands::kNumBands][2] = { { 60.0, 500.0 }, { 2000.0, 4000.0 }, { 14000.0, 20000.0 } };

using Complex = std::complex<double>;

void fft(std::vector<Complex>& a)
{
    const auto n = a.size();

    for (auto i = size_t{ 1 }, j = size_t{ 0 }; i < n; ++i)
    {
        auto bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            std::swap(a[i], a[j]);
    }

    for (auto len = size_t{ 2 }; len <= n; len <<= 1)
    {
        const auto w = std::polar(1.0, -2.0 * 3.14159265358979323846 / static_cast<double>(len));
        for (auto i = size_t{ 0 }; i < n; i += len)
        {
            auto wk = Complex{ 1.0, 0.0 };
            for (auto k = size_t{ 0 }; k < len / 2; ++k, wk *= w)
            {
                const auto u = a[i + k];
                const auto v = a[i + k + len / 2] * wk;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
            }
        }
    }
}

// Energy of 8192 samples from time t, through a Hann window, between two frequencies.
double energyAt(const std::vector<float>& response, double t, const double* range)
{
    const auto length = 8192;
    const auto start = static_cast<int>(t * kSamplingRate);

    std::vector<Complex> spectrum(length);
    for (auto i = 0; i < length; ++i)
    {
        const auto window = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * (i + 0.5) / length);
        spectrum[i] = window * response[start + i];
    }
    fft(spectrum);

    auto sum = 0.0;
    for (auto k = 0; k <= length / 2; ++k)
    {
        const auto frequency = static_cast<double>(k) * kSamplingRate / length;
        if (frequency >= range[0] && frequency < range[1])
            sum += std::norm(spectrum[k]);
    }

    return sum;
}

// The delay lines are drawn with rand() in some versions: every effect gets the same ones, so that two responses differ
// only by their reverb times.
std::vector<float> reverbResponse(const float* reverbTimes, double duration)
{
    srand(1);
    ipl::ReverbEffect effect(ipl::AudioSettings{ kSamplingRate, kFrameSize });

    ipl::Reverb reverb;
    for (auto i = 0; i < ipl::Bands::kNumBands; ++i)
        reverb.reverbTimes[i] = reverbTimes[i];

    ipl::ReverbEffectParams params{};
    params.reverb = &reverb;

    ipl::AudioBuffer in(1, kFrameSize);
    ipl::AudioBuffer out(1, kFrameSize);

    const auto numFrames = static_cast<int>(duration * kSamplingRate) / kFrameSize;
    std::vector<float> response(numFrames * kFrameSize);
    for (auto frame = 0; frame < numFrames; ++frame)
    {
        in.makeSilent();
        if (frame == 0)
            in[0][0] = 1.0f;

        effect.apply(params, in, out);
        std::copy(out[0], out[0] + kFrameSize, &response[frame * kFrameSize]);
    }

    return response;
}

// The parametric part of the hybrid reverb alone (no impulse response).
std::vector<float> hybridTailResponse(const float* reverbTimes, const float* eqCoeffs, double duration)
{
    ipl::HybridReverbEffectSettings settings{};
    settings.numChannels = 1;
    settings.irSize = kFrameSize;
    srand(1);
    ipl::HybridReverbEffect effect(ipl::AudioSettings{ kSamplingRate, kFrameSize }, settings);

    ipl::Reverb reverb;
    for (auto i = 0; i < ipl::Bands::kNumBands; ++i)
        reverb.reverbTimes[i] = reverbTimes[i];

    ipl::HybridReverbEffectDirectParams params{};
    params.reverb = &reverb;
    params.eqCoeffs = eqCoeffs;
    params.numChannels = 1;
    params.numSamples = kFrameSize;

    ipl::AudioBuffer in(1, kFrameSize);
    ipl::AudioBuffer out(1, kFrameSize);

    const auto numFrames = static_cast<int>(duration * kSamplingRate) / kFrameSize;
    std::vector<float> response(numFrames * kFrameSize);
    for (auto frame = 0; frame < numFrames; ++frame)
    {
        in.makeSilent();
        if (frame == 0)
            in[0][0] = 1.0f;

        effect.apply(params, in, out);
        std::copy(out[0], out[0] + kFrameSize, &response[frame * kFrameSize]);
    }

    return response;
}

}

// The reverb of a band, measured in the middle of the band, has to decay as a reverb with that band's reverb time in
// every band, even when the reverb times differ a lot between bands (air absorption, absorbing materials). Second-order
// shelves at the band edges reach well into the neighboring bands: with 2 / 2 / 0.5 s, the shelf at 8 kHz made the
// middle of the middle band decay faster, 3.6 dB under its reverb time after 1 s.
TEST_CASE("The parametric reverb decays at each band's reverb time in the middle of the band, with very different reverb times.", "[ReverbEffect]")
{
    const float cases[][ipl::Bands::kNumBands] = { { 2.0f, 2.0f, 0.5f }, { 0.5f, 2.0f, 2.0f }, { 4.97f, 3.67f, 0.92f } };

    for (const auto& reverbTimes : cases)
    {
        const auto response = reverbResponse(reverbTimes, 2.5);
        const auto minReverbTime = *std::min_element(reverbTimes, reverbTimes + ipl::Bands::kNumBands);

        for (auto band = 0; band < ipl::Bands::kNumBands; ++band)
        {
            // The same reverb time in every band: no band filter at all. The tone correction of a band is
            // sqrt(minimum reverb time / reverb time of the band).
            const float flat[ipl::Bands::kNumBands] = { reverbTimes[band], reverbTimes[band], reverbTimes[band] };
            const auto reference = reverbResponse(flat, 2.5);
            const auto toneCorrection = 10.0 * std::log10(minReverbTime / reverbTimes[band]);

            // Down to 60 dB under the start.
            for (auto t = 0.2; t <= std::min(1.8, static_cast<double>(reverbTimes[band])); t += 0.2)
            {
                const auto difference = 10.0 * std::log10(energyAt(response, t, kMeasuredRanges[band]) /
                                                          energyAt(reference, t, kMeasuredRanges[band])) - toneCorrection;

                INFO("reverb times " << reverbTimes[0] << " / " << reverbTimes[1] << " / " << reverbTimes[2]
                     << ", band " << band << ", at " << t << " s");
                CHECK(std::fabs(difference) < 1.5);
            }
        }
    }
}

// The hybrid reverb gives each band of its parametric part the level of the impulse response at the transition. A band
// that is 40 dB under the others (air absorption at the transition time) has to be 40 dB under them, not 24 dB (the
// floor of the EQ effect), and the shelves of the EQ must not change the other bands.
//
// The tolerance is 1.5 dB because the band shelves are of the 4th order (the 8th order measured 0.07 dB at worst, for
// twice the CPU in the reverb loop). A 4th-order shelf has not finished its step at the edge of a neighboring band's
// measured range: the worst measured error is 1.36 dB (0.92 and 0.90 dB at the other times), in band 0 (60-500 Hz, whose
// upper end is 0.7 octave under the 800 Hz edge) when the EQ sets band 0 to -40 dB and band 1 to 0 dB. Every other band
// and case is within 0.25 dB. The decay test above measured 0.13 dB at worst with this order.
TEST_CASE("The parametric part of the hybrid reverb has the level of each band.", "[HybridReverb]")
{
    const float reverbTimes[ipl::Bands::kNumBands] = { 1.0f, 1.0f, 1.0f };
    const float flat[ipl::Bands::kNumBands] = { 1.0f, 1.0f, 1.0f };
    const auto reference = hybridTailResponse(reverbTimes, flat, 1.0);

    const float cases[][ipl::Bands::kNumBands] = { { 1.0f, 1.0f, 0.01f }, { 1.0f, 0.1f, 0.01f }, { 0.01f, 1.0f, 0.1f } };

    for (const auto& eqCoeffs : cases)
    {
        const auto response = hybridTailResponse(reverbTimes, eqCoeffs, 1.0);

        for (auto band = 0; band < ipl::Bands::kNumBands; ++band)
        {
            for (auto t = 0.2; t <= 0.61; t += 0.2)
            {
                const auto level = 10.0 * std::log10(energyAt(response, t, kMeasuredRanges[band]) /
                                                     energyAt(reference, t, kMeasuredRanges[band]));
                const auto expected = 20.0 * std::log10(eqCoeffs[band]);

                INFO("EQ " << eqCoeffs[0] << " / " << eqCoeffs[1] << " / " << eqCoeffs[2] << ", band " << band
                     << ", at " << t << " s");
                CHECK(std::fabs(level - expected) < 1.5);
            }
        }
    }
}

// The hybrid reverb's level in a band must not depend on the reverb times of the other bands.
TEST_CASE("The parametric part of the hybrid reverb keeps the level of a band when the other bands ring longer.", "[HybridReverb]")
{
    const float eqCoeffs[ipl::Bands::kNumBands] = { 1.0f, 1.0f, 0.03f };
    const float tilted[ipl::Bands::kNumBands] = { 4.97f, 3.67f, 0.92f };
    const float flat[ipl::Bands::kNumBands] = { 0.92f, 0.92f, 0.92f };

    const auto response = hybridTailResponse(tilted, eqCoeffs, 1.2);
    const auto reference = hybridTailResponse(flat, eqCoeffs, 1.2);

    for (auto t = 0.2; t <= 0.81; t += 0.2)
    {
        const auto difference = 10.0 * std::log10(energyAt(response, t, kMeasuredRanges[2]) /
                                                  energyAt(reference, t, kMeasuredRanges[2]));

        INFO("at " << t << " s");
        CHECK(std::fabs(difference) < 1.5);
    }
}

#endif
