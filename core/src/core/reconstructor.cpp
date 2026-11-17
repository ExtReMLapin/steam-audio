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

#include "reconstructor.h"

#include <cstring>
#include <functional>
#include <random>

#include "array_math.h"
#include "fft.h"
#include "log.h"
#include "math_functions.h"
#include "profiler.h"
#include "propagation_medium.h"
#include "sh.h"

namespace ipl {

// --------------------------------------------------------------------------------------------------------------------
// IReconstructor
// --------------------------------------------------------------------------------------------------------------------

const float IReconstructor::kEnergyThreshold = 1e-7f;
const float IReconstructor::kMinVariance = 1e-5f;


// --------------------------------------------------------------------------------------------------------------------
// Reconstructor
// --------------------------------------------------------------------------------------------------------------------

Reconstructor::Reconstructor(float maxDuration,
                             int maxOrder,
                             int samplingRate)
    : mMaxDuration(maxDuration)
    , mMaxOrder(maxOrder)
    , mSamplingRate(samplingRate)
    , mWhiteNoise(Bands::kNumBands, static_cast<int>(ceilf(maxDuration * samplingRate)))
    , mBandIRs(Bands::kNumBands, static_cast<int>(ceilf(maxDuration * samplingRate)))
{
    // One generator shared by all bands, so that each band gets a different part of the sequence. The same noise in
    // every band is summed coherently by the band filters, which leaves a dip of up to 7 dB around 800 Hz and 8 kHz,
    // where adjacent bands overlap. std::mt19937 rather than std::default_random_engine, which is a different
    // generator in each standard library (and with some of them, consecutive seeds give correlated sequences).
    std::mt19937 randomGenerator;
    std::uniform_real_distribution<float> uniformDistribution(-1.0f, 1.0f);
    auto uniformRandom = std::bind(uniformDistribution, std::ref(randomGenerator));

    // The noise of each band is band-limited here, once, rather than the response of each band being filtered after
    // it is shaped by the energy field. The envelope varies over bins of 10 ms, so shaping band-limited noise leaves
    // its spectrum in the band. Band filters applied after the envelope (second-order, 6 dB per octave) let a band that
    // decays slowly leak into a band that decays fast: once air absorption makes the highest band decay much faster than
    // the middle one, the highs of the response decay at the rate of the middle band.
    //
    // The bands get independent noise, so the response of a flat energy field is flat if the squared magnitudes of the
    // band filters add up to 1. Each band edge is a crossover of a third of an octave, raised cosine in log frequency,
    // and nothing of a band reaches past it. The filters have zero phase, and the noise is filtered circularly, so
    // there is no transient at either end.
    FFT fft(static_cast<int>(mWhiteNoise.size(1)));
    Array<float> noise(fft.numRealSamples);
    Array<complex_t> spectrum(fft.numComplexSamples);

    auto crossover = [](float frequency, float edgeFrequency)
    {
        const auto kHalfWidth = 1.0f / 6.0f; // octaves
        if (frequency <= 0.0f)
            return 0.0f;

        auto octaves = log2f(frequency / edgeFrequency);
        if (octaves <= -kHalfWidth)
            return 0.0f;
        if (octaves >= kHalfWidth)
            return 1.0f;

        return 0.5f - 0.5f * cosf(Math::kPi * (octaves + kHalfWidth) / (2.0f * kHalfWidth));
    };

    for (auto i = 0; i < Bands::kNumBands; ++i)
    {
        for (auto j = 0; j < fft.numRealSamples; ++j)
        {
            noise[j] = uniformRandom();
        }

        fft.applyForward(noise.data(), spectrum.data());

        for (auto k = 0; k < fft.numComplexSamples; ++k)
        {
            auto frequency = (static_cast<float>(k) * samplingRate) / fft.numRealSamples;
            auto above = (i == 0) ? 1.0f : crossover(frequency, Bands::kLowCutoffFrequencies[i]);
            auto aboveNext = (i == Bands::kNumBands - 1) ? 0.0f : crossover(frequency, Bands::kLowCutoffFrequencies[i + 1]);
            spectrum[k] *= sqrtf(std::max(above - aboveNext, 0.0f));
        }

        fft.applyInverse(spectrum.data(), noise.data());

        memcpy(mWhiteNoise[i], noise.data(), mWhiteNoise.size(1) * sizeof(float));
    }
}

void Reconstructor::reconstruct(int numIRs,
                                const EnergyField* const* energyFields,
                                const float* const* distanceAttenuationCorrectionCurves,
                                const AirAbsorptionModel* airAbsorptionModels,
                                ImpulseResponse* const* impulseResponses,
                                ReconstructionType type,
                                float duration,
                                int order)
{
    PROFILE_FUNCTION();

    for (auto i = 0; i < numIRs; ++i)
    {
        const auto& energyField = *energyFields[i];
        auto& impulseResponse = *impulseResponses[i];

        auto numChannels = SphericalHarmonics::numCoeffsForOrder(order);
        auto numSamples = static_cast<int>(ceilf(duration * mSamplingRate));

        numChannels = std::min({ energyField.numChannels(), impulseResponse.numChannels(), numChannels });
        auto numSamplesPerBin = static_cast<int>(ceilf(EnergyField::kBinDuration * mSamplingRate));
        numSamples = std::min({ impulseResponse.numSamples(), numSamples });
        auto numBins = std::min({ energyField.numBins(), static_cast<int>(ceilf(static_cast<float>(numSamples) / static_cast<float>(numSamplesPerBin))) });

        impulseResponses[i]->reset();

        for (auto iChannel = 0; iChannel < numChannels; ++iChannel)
        {
            for (auto iBand = 0; iBand < Bands::kNumBands; ++iBand)
            {
                for (auto iBin = 0; iBin < numBins; ++iBin)
                {
                    auto numBinSamples = std::min(numSamplesPerBin, numSamples - iBin * numSamplesPerBin);
                    auto normalization = 1.0f;

                    if (type == ReconstructionType::Linear)
                    {
                        auto energy = 0.0f;
                        if (fabsf(energyField[iChannel][iBand][iBin]) >= kEnergyThreshold && fabsf(energyField[0][iBand][iBin]) >= kEnergyThreshold)
                        {
                            energy = energyField[iChannel][iBand][iBin] / sqrtf(energyField[0][iBand][iBin] * sqrtf(4.0f * Math::kPi));
                        }

                        auto prevEnergy = 0.0f;
                        if (iBin == 0)
                        {
                            prevEnergy = energy;
                        }
                        else if (fabsf(energyField[iChannel][iBand][iBin - 1]) >= kEnergyThreshold && fabsf(energyField[0][iBand][iBin - 1]) >= kEnergyThreshold)
                        {
                            prevEnergy = energyField[iChannel][iBand][iBin - 1] / sqrtf(energyField[0][iBand][iBin - 1] * sqrtf(4.0f * Math::kPi));
                        }

                        for (auto iBinSample = 0, iSample = iBin * numSamplesPerBin; iBinSample < numBinSamples; ++iBinSample, ++iSample)
                        {
                            auto weight = static_cast<float>(iBinSample) / static_cast<float>(numSamplesPerBin);
                            auto sampleEnergy = (1.0f - weight) * prevEnergy + weight * energy;

                            mBandIRs[iBand][iSample] = sampleEnergy * mWhiteNoise[iBand][iSample];
                        }
                    }
                    else if (type == ReconstructionType::Gaussian)
                    {
                        if (fabsf(energyField[iChannel][iBand][iBin]) < kEnergyThreshold || fabsf(energyField[0][iBand][iBin]) < kEnergyThreshold)
                            continue;

                        auto tMean = ((iBin + 0.5f) * numSamplesPerBin) / mSamplingRate;
                        auto tVariance = kMinVariance;

                        auto iSample = iBin * numSamplesPerBin;

                        auto t = iSample / static_cast<float>(mSamplingRate);
                        auto dt = 1.0f / mSamplingRate;
                        auto g = expf(-((t - tMean) * (t - tMean)) / (2.0f * tVariance));
                        auto dg = expf(-(dt * ((2.0f * (t - tMean)) + dt)) / (2.0f * tVariance));
                        auto ddg = expf(-(dt * dt) / tVariance);

                        for (auto iBinSample = 0; iBinSample < numBinSamples; ++iBinSample, ++iSample)
                        {
                            mBandIRs[iBand][iSample] = g * mWhiteNoise[iBand][iSample];

                            g *= dg;
                            dg *= ddg;
                        }

                        normalization = energyField[iChannel][iBand][iBin] / sqrtf(energyField[0][iBand][iBin] * sqrtf(4.0f * Math::kPi));
                        assert(Math::isFinite(normalization));
                    }

                    // 0.5 is for sqrt
                    normalization *= airAbsorptionModels[i].evaluate(0.5f * PropagationMedium::kSpeedOfSound * ((iBin + 0.5f) * numSamplesPerBin * (1.0f / mSamplingRate)), iBand);

                    auto iSample = iBin * numSamplesPerBin;
                    ArrayMath::scale(numBinSamples, &mBandIRs[iBand][iSample], normalization, &mBandIRs[iBand][iSample]);
                }
            }

            // The noise of each band is band-limited already.
            memcpy(impulseResponse[iChannel], mBandIRs[0], numSamples * sizeof(float));
            for (auto iBand = 1; iBand < Bands::kNumBands; ++iBand)
            {
                ArrayMath::add(numSamples, impulseResponse[iChannel], mBandIRs[iBand], impulseResponse[iChannel]);
            }

            if (distanceAttenuationCorrectionCurves[i])
            {
                ArrayMath::multiply(numSamples, impulseResponse[iChannel], distanceAttenuationCorrectionCurves[i], impulseResponse[iChannel]);
            }
        }
    }
}

}
