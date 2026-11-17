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

#include <cmath>
#include <complex>
#include <utility>
#include <vector>

#include <air_absorption.h>
#include <distance_attenuation.h>
#include <energy_field.h>
#include <impulse_response.h>
#include <propagation_medium.h>
#include <reconstructor.h>

#if !defined(IPL_ENABLE_OCTAVE_BANDS)

namespace {

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

const auto kSamplingRate = 48000;
const auto kDuration = 1.0f;

// Energy of the omni channel between two frequencies.
double bandEnergy(const ipl::ImpulseResponse& ir, double lowFrequency, double highFrequency)
{
    auto n = size_t{ 1 };
    while (n < static_cast<size_t>(ir.numSamples()))
        n <<= 1;

    std::vector<Complex> spectrum(n, Complex{ 0.0, 0.0 });
    for (auto i = 0; i < ir.numSamples(); ++i)
        spectrum[i] = ir[0][i];
    fft(spectrum);

    auto sum = 0.0;
    for (auto k = size_t{ 0 }; k <= n / 2; ++k)
    {
        const auto frequency = static_cast<double>(k) * kSamplingRate / static_cast<double>(n);
        if (frequency >= lowFrequency && frequency < highFrequency)
            sum += std::norm(spectrum[k]);
    }

    return sum;
}

// Reflections that arrive between firstBin and firstBin + 10 bins after the direct sound, from a source at
// directDistance.
double reflectionEnergy(int firstBin, float directDistance, const ipl::AirAbsorptionModel& airAbsorption,
                        double lowFrequency, double highFrequency)
{
    ipl::EnergyField energyField(kDuration, 0);
    for (auto iBand = 0; iBand < ipl::Bands::kNumBands; ++iBand)
    {
        for (auto iBin = firstBin; iBin < firstBin + 10; ++iBin)
        {
            energyField[0][iBand][iBin] = 1.0f;
        }
    }

    ipl::Reconstructor reconstructor(kDuration, 0, kSamplingRate);
    ipl::ImpulseResponse ir(kDuration, 0, kSamplingRate);

    const ipl::EnergyField* energyFields[] = { &energyField };
    const float* distanceAttenuationCorrectionCurves[] = { nullptr };
    ipl::ImpulseResponse* impulseResponses[] = { &ir };
    const float directDistances[] = { directDistance };

    reconstructor.reconstruct(1, energyFields, distanceAttenuationCorrectionCurves, &airAbsorption, impulseResponses,
                              ipl::ReconstructionType::Linear, kDuration, 0, directDistances);

    return bandEnergy(ir, lowFrequency, highFrequency);
}

}

// The energy field holds the delay of a reflection after the direct sound, so the air absorption of the reflections
// left out the distance between the source and the listener, which every reflection travels too: the highs of the
// reflections of a source 100 m away were as bright as those of a source next to the listener. A reflection that
// arrives t after the direct sound of a source at a distance D has travelled D + c t, and it has to be absorbed as
// much as a reflection that arrives D / c + t after the direct sound of a source at the listener.
TEST_CASE("The air absorption of reflections counts the distance between the source and the listener.", "[Reconstructor]")
{
    const ipl::AirAbsorptionModel airAbsorption;

    float noAbsorptionCoefficients[ipl::Bands::kNumBands] = {};
    const ipl::AirAbsorptionModel noAbsorption(noAbsorptionCoefficients, nullptr, nullptr);

    const double ranges[][2] = { { 2000.0, 5000.0 }, { 12000.0, 18000.0 } };

    // 0.2 s and 0.5 s of travel: 69 m and 172 m.
    for (auto delayBins : { 20, 50 })
    {
        const auto directDistance = ipl::PropagationMedium::kSpeedOfSound * delayBins * ipl::EnergyField::kBinDuration;

        for (const auto& range : ranges)
        {
            const auto absorbed = reflectionEnergy(5, directDistance, airAbsorption, range[0], range[1]) /
                                  reflectionEnergy(5, directDistance, noAbsorption, range[0], range[1]);

            const auto expected = reflectionEnergy(5 + delayBins, 0.0f, airAbsorption, range[0], range[1]) /
                                  reflectionEnergy(5 + delayBins, 0.0f, noAbsorption, range[0], range[1]);

            const auto difference = 10.0 * std::log10(absorbed / expected);

            INFO("source at " << directDistance << " m, " << range[0] << " to " << range[1] << " Hz");
            CHECK(std::fabs(difference) < 0.3);
        }
    }
}

// The distance attenuation correction curve goes from the default distance attenuation model, which the energy field
// is simulated with (over the whole path), to the source's model: it has to be evaluated over the whole path too.
TEST_CASE("The distance attenuation correction curve counts the distance between the source and the listener.", "[DistanceAttenuation]")
{
    auto inverseSquare = [](float distance, void*)
    {
        return 1.0f / std::max(distance * distance, 1.0f);
    };

    const ipl::DistanceAttenuationModel defaultModel{};
    const ipl::DistanceAttenuationModel inverseSquareModel(1.0f, inverseSquare, nullptr);

    const auto numSamples = kSamplingRate / 2;
    std::vector<float> curve(numSamples);

    for (auto directDistance : { 0.0f, 10.0f, 100.0f })
    {
        ipl::DistanceAttenuationModel::generateCorrectionCurve(defaultModel, inverseSquareModel, kSamplingRate,
                                                               numSamples, curve.data(), directDistance);

        for (auto i = 0; i < numSamples; i += 2400)
        {
            const auto distance = directDistance + ipl::PropagationMedium::kSpeedOfSound * i / static_cast<float>(kSamplingRate);
            const auto expected = inverseSquareModel.evaluate(distance) / defaultModel.evaluate(distance);

            INFO("source at " << directDistance << " m, sample " << i);
            CHECK(curve[i] == Approx(expected).epsilon(1e-4));
        }
    }
}

#endif

#include <hybrid_reverb_estimator.h>

#if !defined(IPL_ENABLE_OCTAVE_BANDS)

// The hybrid reverb takes the level of each band at the transition from the energy field, absorbed by the air: over
// the distance between the source and the listener too, like the reflections before it.
TEST_CASE("The hybrid reverb EQ counts the distance between the source and the listener in its air absorption.", "[HybridReverbEstimator]")
{
    const auto samplingRate = 48000;
    const auto duration = 1.0f;
    const auto transitionTime = 0.5f;
    const auto overlapFraction = 0.25f;

    ipl::EnergyField energyField(duration, 0);
    for (auto bin = 0; bin < energyField.numBins(); ++bin)
    {
        for (auto band = 0; band < ipl::Bands::kNumBands; ++band)
        {
            energyField[0][band][bin] = expf(-bin * ipl::EnergyField::kBinDuration / 0.2f);
        }
    }

    ipl::Reverb reverb;
    for (auto band = 0; band < ipl::Bands::kNumBands; ++band)
    {
        reverb.reverbTimes[band] = 1.0f;
    }

    ipl::HybridReverbEstimator estimator(duration, samplingRate, 1024);
    const ipl::AirAbsorptionModel airAbsorption;

    auto estimateEQ = [&](float directDistance, float* eq)
    {
        ipl::ImpulseResponse ir(duration, 0, samplingRate);
        auto delay = 0;
        estimator.estimate(&energyField, reverb, airAbsorption, ir, transitionTime, overlapFraction, 0, eq, delay,
                           directDistance);
    };

    float nearEQ[ipl::Bands::kNumBands];
    estimateEQ(0.0f, nearEQ);

    const auto directDistance = 100.0f;
    float farEQ[ipl::Bands::kNumBands];
    estimateEQ(directDistance, farEQ);

    // The cutoff bin is the same for both, so the ratio is the absorption over the direct distance alone.
    for (auto band = 0; band < ipl::Bands::kNumBands; ++band)
    {
        INFO("band " << band);
        CHECK(farEQ[band] / nearEQ[band] == Approx(airAbsorption.evaluate(directDistance, band)).epsilon(1e-3));
    }
}

#endif
