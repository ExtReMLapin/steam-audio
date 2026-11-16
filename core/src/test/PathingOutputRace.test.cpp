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
#include <atomic>
#include <thread>
#include <scene_factory.h>
#include <simulation_data.h>
#include <simulation_manager.h>
using namespace ipl;

TEST_CASE("Pathing outputs read while a simulation writes them are never torn", "[PathingRace]")
{
    shared_ptr<IScene> scene(SceneFactory::create(SceneType::Default, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
    scene->commit();
    SimulationManager manager(false, false, true, SceneType::Default, IndirectEffectType::Parametric, 32, 1024, 32,
                              1.0f, 3, 4, 1, 1, 1, 8, true, Vector3f(0, -1, 0), 48000, 1024, nullptr, nullptr, nullptr);
    manager.scene() = scene;
    auto probes = make_shared<ProbeBatch>();
    probes->addProbe(Sphere(Vector3f(0, 0, 0), 2.0f));
    probes->commit();
    manager.addProbeBatch(probes);
    auto source = make_shared<SimulationData>(false, true, SceneType::Default, IndirectEffectType::Parametric, 32, 1.0f, 3, 48000, 1024, nullptr, nullptr);
    source->pathingInputs.enabled = true;
    source->pathingInputs.probes = probes;
    source->pathingInputs.visRange = 100.0f;
    source->pathingInputs.order = 3;
    manager.addSource(source);
    manager.commit();
    SharedPathingSimulationInputs shared;
    shared.listener = CoordinateSpace3f(Vector3f(0, 0, 0));
    manager.setSharedPathingInputs(shared);

    // Direct path, gain 1/d: sum of squares of N3D SH coefficients of order 3 is 16 * gain^2.
    const float d[2] = { 2.0f, 8.0f };
    std::atomic<bool> stop{false};
    std::thread writer([&] {
        int i = 0;
        while (!stop) {
            source->pathingInputs.source = CoordinateSpace3f(Vector3f(d[i & 1], 0, 0));
            manager.simulatePathing();
            ++i;
        }
    });
    long torn = 0, reads = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(3)) {
        float eq[Bands::kNumBands];
        const float* sh = nullptr;
        source->readPathingOutputs(eq, &sh);   // what iplSourceGetOutputs hands out
        float e = 0; for (int k = 0; k < 16; ++k) e += sh[k] * sh[k];
        ++reads;
        if (e == 0.0f) continue;
        bool ok = false;
        for (int s = 0; s < 2; ++s) { float g = 1.0f / d[s]; if (fabsf(e - (16.0f / (4.0f * 3.14159265f)) * g * g) < 1e-3f * (16.0f / (4.0f * 3.14159265f)) * g * g) ok = true; }
        if (!ok) ++torn;
    }
    stop = true; writer.join();
    printf("reads %ld, torn %ld\n", reads, torn);
    REQUIRE(torn == 0);
}
