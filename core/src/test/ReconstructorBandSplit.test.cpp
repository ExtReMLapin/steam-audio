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
#include <energy_field.h>
#include <impulse_response.h>
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

// Energy of ir[start, start + length) between two frequencies, through a Hann window.
double windowEnergy(const float* ir, int start, int length, int samplingRate, double lowFrequency, double highFrequency)
{
    auto n = size_t{ 1 };
    while (n < static_cast<size_t>(length))
        n <<= 1;

    std::vector<Complex> spectrum(n, Complex{ 0.0, 0.0 });
    for (auto i = 0; i < length; ++i)
    {
        const auto window = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * (i + 0.5) / length);
        spectrum[i] = window * ir[start + i];
    }
    fft(spectrum);

    auto sum = 0.0;
    for (auto k = size_t{ 0 }; k <= n / 2; ++k)
    {
        const auto frequency = static_cast<double>(k) * samplingRate / static_cast<double>(n);
        if (frequency >= lowFrequency && frequency < highFrequency)
            sum += std::norm(spectrum[k]);
    }

    return sum;
}

// An energy field whose bands decay at the given rates (dB/s), from the same level (high enough for no bin to fall
// under the energy threshold of the reconstructor within a second). A rate below zero leaves the band empty.
void fillDecay(ipl::EnergyField& energyField, const float* decayRates)
{
    for (auto iBand = 0; iBand < ipl::Bands::kNumBands; ++iBand)
    {
        for (auto iBin = 0; iBin < energyField.numBins(); ++iBin)
        {
            const auto t = (iBin + 0.5f) * ipl::EnergyField::kBinDuration;
            energyField[0][iBand][iBin] = (decayRates[iBand] < 0.0f) ? 0.0f : 1e3f * powf(10.0f, -decayRates[iBand] * t / 10.0f);
        }
    }
}

void reconstruct(ipl::Reconstructor& reconstructor, const ipl::EnergyField& energyField, ipl::ImpulseResponse& ir,
                 ipl::ReconstructionType type, float duration)
{
    float noAbsorption[ipl::Bands::kNumBands] = {};
    ipl::AirAbsorptionModel airAbsorption(noAbsorption, nullptr, nullptr);

    const ipl::EnergyField* energyFields[] = { &energyField };
    const float* distanceAttenuationCorrectionCurves[] = { nullptr };
    ipl::ImpulseResponse* impulseResponses[] = { &ir };

    reconstructor.reconstruct(1, energyFields, distanceAttenuationCorrectionCurves, &airAbsorption, impulseResponses,
                              type, duration, 0);
}

}

// When one band of the energy field decays much faster than the band below it (air absorption, absorbing materials),
// the content of the response in the upper band has to decay at the upper band's rate. With band filters of 6 dB per
// octave, the lower band's skirt sits 10 to 20 dB under the upper band at first, then takes over the upper band's
// frequencies once the upper band has decayed more than that: the highs of the response decay at the rate of the mids.
TEST_CASE("Each band of a reconstructed response follows the decay of its own band.", "[Reconstructor]")
{
    const auto duration = 1.0f;

    struct Case
    {
        float decayRates[ipl::Bands::kNumBands];   // dB/s of the full field
        int band;                                  // the band whose decay is checked
        double lowFrequency;                       // where it is measured
        double highFrequency;
    };

    // 60 dB/s between adjacent bands: 48 dB apart at 0.8 s.
    const Case cases[] = {
        { { 40.0f, 40.0f, 100.0f }, 2, 10000.0, 20000.0 },
        { { 40.0f, 100.0f, 100.0f }, 1, 2000.0, 5000.0 },
    };

    for (auto samplingRate : { 44100, 48000, 96000 })
    {
        ipl::Reconstructor reconstructor(duration, 0, samplingRate);
        ipl::ImpulseResponse full(duration, 0, samplingRate);
        ipl::ImpulseResponse alone(duration, 0, samplingRate);

        for (const auto& testCase : cases)
        {
            ipl::EnergyField fullField(duration, 0);
            fillDecay(fullField, testCase.decayRates);

            float aloneRates[ipl::Bands::kNumBands];
            for (auto iBand = 0; iBand < ipl::Bands::kNumBands; ++iBand)
                aloneRates[iBand] = (iBand == testCase.band) ? testCase.decayRates[iBand] : -1.0f;

            ipl::EnergyField aloneField(duration, 0);
            fillDecay(aloneField, aloneRates);

            for (auto type : { ipl::ReconstructionType::Linear, ipl::ReconstructionType::Gaussian })
            {
                reconstruct(reconstructor, fullField, full, type, duration);
                reconstruct(reconstructor, aloneField, alone, type, duration);

                for (auto start = 0.3; start < 0.75; start += 0.1)
                {
                    const auto first = static_cast<int>(start * samplingRate);
                    const auto length = static_cast<int>(0.1 * samplingRate);

                    const auto fullEnergy = windowEnergy(full[0], first, length, samplingRate, testCase.lowFrequency, testCase.highFrequency);
                    const auto aloneEnergy = windowEnergy(alone[0], first, length, samplingRate, testCase.lowFrequency, testCase.highFrequency);
                    const auto difference = 10.0 * std::log10(fullEnergy / aloneEnergy);

                    INFO("sampling rate " << samplingRate << ", band " << testCase.band << ", type "
                         << static_cast<int>(type) << ", window at " << start << " s");
                    CHECK(std::fabs(difference) < 1.5);
                }
            }
        }
    }
}

// With independent noise in each band, the response of a flat energy field is flat if the squared magnitudes of the
// band filters add up to 1, at every sampling rate.
TEST_CASE("A flat energy field is reconstructed flat at every sampling rate.", "[Reconstructor]")
{
    const auto duration = 4.0f;

    ipl::EnergyField energyField(duration, 0);
    const float flat[ipl::Bands::kNumBands] = {};
    fillDecay(energyField, flat);

    for (auto samplingRate : { 44100, 48000, 96000 })
    {
        ipl::Reconstructor reconstructor(duration, 0, samplingRate);
        ipl::ImpulseResponse ir(duration, 0, samplingRate);
        reconstruct(reconstructor, energyField, ir, ipl::ReconstructionType::Linear, duration);

        const auto numSamples = ir.numSamples();
        auto bandEnergy = [&](double center)
        {
            const auto low = center * std::pow(2.0, -1.0 / 6.0);
            const auto high = center * std::pow(2.0, 1.0 / 6.0);
            return windowEnergy(ir[0], 0, numSamples, samplingRate, low, high) / (high - low);
        };

        auto reference = 0.0;
        for (auto frequency : { 125.0, 157.0, 198.0, 250.0 })
            reference += bandEnergy(frequency) / 4.0;

        for (auto i = -5; i <= 12; ++i)
        {
            const auto frequency = 1000.0 * std::pow(2.0, i / 3.0);
            const auto level = 10.0 * std::log10(bandEnergy(frequency) / reference);

            INFO("sampling rate " << samplingRate << ", third octave around " << frequency << " Hz");
            CHECK(std::fabs(level) < 1.0);
        }
    }
}

#endif
