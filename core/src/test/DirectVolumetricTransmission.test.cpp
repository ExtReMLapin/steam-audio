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

#include <direct_simulator.h>
#include <scene_factory.h>
using namespace ipl;

TEST_CASE("Transmission applies to the occluded part of a partially occluded source", "[DirectSimulator]")
{
    // A wall at x = 5 covers y > 0.05, so it blocks part of the source's volume (a sphere of radius 0.5 around
    // x = 10), but not the line from the listener to the source's center.
    const auto wallTransmission = 0.05f;
    Vector3f vertices[] = {
        Vector3f(5.0f, 0.05f, -50.0f), Vector3f(5.0f, 50.0f, -50.0f), Vector3f(5.0f, 50.0f, 50.0f), Vector3f(5.0f, 0.05f, 50.0f)
    };
    Triangle triangles[] = { Triangle{ { 0, 1, 2 } }, Triangle{ { 0, 2, 3 } } };
    int materialIndices[] = { 0, 0 };
    Material material{};
    for (auto i = 0; i < Bands::kNumBands; ++i)
    {
        material.absorption[i] = 0.1f;
        material.transmission[i] = wallTransmission;
    }
    material.scattering = 0.5f;

    auto scene = shared_ptr<IScene>(SceneFactory::create(SceneType::Default, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
    scene->addStaticMesh(scene->createStaticMesh(4, 2, 1, vertices, triangles, materialIndices, &material));
    scene->commit();

    DirectSimulator simulator(64);
    DirectSoundPath path{};
    simulator.simulate(scene.get(), static_cast<DirectSimulationFlags>(CalcOcclusion | CalcTransmission),
                       CoordinateSpace3f{ -Vector3f::kZAxis, Vector3f::kYAxis, Vector3f(10.0f, 0.0f, 0.0f) },
                       CoordinateSpace3f{ -Vector3f::kZAxis, Vector3f::kYAxis, Vector3f::kZero }, DistanceAttenuationModel{},
                       AirAbsorptionModel{}, Directivity{}, OcclusionType::Volumetric, 0.5f, 64, 1, path);

    REQUIRE(path.occlusion > 0.1f);
    REQUIRE(path.occlusion < 0.9f);

    // The occluded part of the sound goes through the wall, so it is attenuated by the wall's transmission. This is the
    // gain that the direct effect applies.
    auto gain = path.occlusion + (1.0f - path.occlusion) * path.transmission[1];
    auto expected = path.occlusion + (1.0f - path.occlusion) * wallTransmission;
    INFO("occlusion " << path.occlusion << ", transmission " << path.transmission[1] << ", gain " << gain
                      << ", expected " << expected);
    CHECK(gain == Approx(expected).epsilon(0.02));
}
