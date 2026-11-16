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
#include "log.h"
#include "math_functions.h"
#include "profiler.h"
#include "propagation_medium.h"
#include "sh.h"

namespace ipl {

// --------------------------------------------------------------------------------------------------------------------
// IReconstructor
// --------------------------------------------------------------------------------------------------------------------

// The energy field is in units of a source of power 1: a source far away, or a late reflection, has an energy that is
// small and still has to be heard. Only a bin that holds nothing (or less than a float keeps) is left out.
const float IReconstructor::kEnergyThreshold = 1e-30f;
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
    , mFilters(Bands::kNumBands)
{
    // One generator shared by all bands, so that each band gets a different part of the sequence. The same noise in
    // every band is summed coherently by the band filters, which leaves a dip of up to 7 dB around 800 Hz and 8 kHz,
    // where adjacent bands overlap. std::mt19937 rather than std::default_random_engine, which is a different
    // generator in each standard library (and with some of them, consecutive seeds give correlated sequences).
    std::mt19937 randomGenerator;
    std::uniform_real_distribution<float> uniformDistribution(-1.0f, 1.0f);
    auto uniformRandom = std::bind(uniformDistribution, std::ref(randomGenerator));

    for (auto i = 0; i < Bands::kNumBands; ++i)
    {
        for (auto j = 0u; j < mWhiteNoise.size(1); ++j)
        {
            mWhiteNoise[i][j] = uniformRandom();
        }
    }

    IIR filters[Bands::kNumBands];
    IIR::bandFilters(filters, samplingRate);
    for (auto i = 0; i < Bands::kNumBands; ++i)
    {
        mFilters[i].setFilter(filters[i]);
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
                // Bins that are skipped, or beyond the end of the energy field, must not keep the data of a previous
                // channel or reconstruction.
                memset(mBandIRs[iBand], 0, numSamples * sizeof(float));

                for (auto iBin = 0; iBin < numBins; ++iBin)
                {
                    auto numBinSamples = std::min(numSamplesPerBin, numSamples - iBin * numSamplesPerBin);
                    auto normalization = 1.0f;

                    if (type == ReconstructionType::Linear)
                    {
                        // The energy field holds the energy that arrives in a bin, in intensity units (1 / (4 pi d^2) for
                        // a source of power 1). The noise of the bin has to carry that energy over its samples, times
                        // 4 pi so that the response is in the units of the direct sound (an amplitude of 1 / d), where the
                        // omni channel is Y00 times the amplitude. Without this the noise had the energy of a bin in each
                        // of its samples: 13 times too much at 48 kHz, and in proportion to the sampling rate.
                        normalization = sqrtf(4.0f * Math::kPi / (kNoiseVariance * static_cast<float>(numSamplesPerBin)));

                        // The amplitude of a bin is placed at the middle of the bin (as the Gaussian reconstruction and the
                        // air absorption do), and linearly interpolated towards the middle of the neighboring bins.
                        auto binAmplitude = [&](int bin)
                        {
                            if (fabsf(energyField[iChannel][iBand][bin]) >= kEnergyThreshold && fabsf(energyField[0][iBand][bin]) >= kEnergyThreshold)
                            {
                                return energyField[iChannel][iBand][bin] / sqrtf(energyField[0][iBand][bin] * sqrtf(4.0f * Math::kPi));
                            }

                            return 0.0f;
                        };

                        auto energy = binAmplitude(iBin);
                        auto prevEnergy = (iBin == 0) ? energy : binAmplitude(iBin - 1);
                        auto nextEnergy = (iBin + 1 < energyField.numBins()) ? binAmplitude(iBin + 1) : energy;

                        for (auto iBinSample = 0, iSample = iBin * numSamplesPerBin; iBinSample < numBinSamples; ++iBinSample, ++iSample)
                        {
                            auto position = (static_cast<float>(iBinSample) + 0.5f) / static_cast<float>(numSamplesPerBin);
                            auto sampleEnergy = 0.0f;
                            if (position < 0.5f)
                            {
                                auto weight = position + 0.5f;
                                sampleEnergy = (1.0f - weight) * prevEnergy + weight * energy;
                            }
                            else
                            {
                                auto weight = position - 0.5f;
                                sampleEnergy = (1.0f - weight) * energy + weight * nextEnergy;
                            }

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

                    // The air absorption model gives an amplitude gain, like the normalization.
                    normalization *= airAbsorptionModels[i].evaluate(PropagationMedium::kSpeedOfSound * ((iBin + 0.5f) * numSamplesPerBin * (1.0f / mSamplingRate)), iBand);

                    auto iSample = iBin * numSamplesPerBin;
                    ArrayMath::scale(numBinSamples, &mBandIRs[iBand][iSample], normalization, &mBandIRs[iBand][iSample]);
                }
            }

            for (auto iBand = 0; iBand < Bands::kNumBands; ++iBand)
            {
                mFilters[iBand].reset();
            }

            mFilters[0].apply(numSamples, mBandIRs[0], impulseResponse[iChannel]);
            for (auto iBand = 1; iBand < Bands::kNumBands; ++iBand)
            {
                mFilters[iBand].apply(numSamples, mBandIRs[iBand], mBandIRs[iBand]);
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
