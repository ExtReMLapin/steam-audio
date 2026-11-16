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
#include <cmath>
#include <map>
#include <vector>

#include <air_absorption.h>
#include <embree_device.h>
#include <energy_field.h>
#include <job_graph.h>
#include <reflection_simulator_factory.h>
#include <reverb_estimator.h>
#include <scene_factory.h>

using namespace ipl;

namespace {

// Simulates reflections in a closed, uniformly absorbing box (the source and listener are inside it) and returns
// the mid-band reverberation time that Steam Audio's own estimator reads from the resulting energy field.
float simulatedReverbTime(SceneType sceneType,
                          float sizeX,
                          float sizeY,
                          float sizeZ,
                          float absorption,
                          float scattering,
                          int numRays,
                          int numBounces,
                          float duration)
{
    auto embree = (sceneType == SceneType::Embree) ? make_shared<EmbreeDevice>() : nullptr;
    auto scene = shared_ptr<IScene>(SceneFactory::create(sceneType, nullptr, nullptr, nullptr, nullptr, nullptr, embree, nullptr));

    Material material;
    for (auto i = 0; i < Bands::kNumBands; ++i)
    {
        material.absorption[i] = absorption;
        material.transmission[i] = 0.0f;
    }
    material.scattering = scattering;

    Vector3f vertices[8];
    for (auto i = 0; i < 8; ++i)
    {
        vertices[i] = Vector3f((i & 1) ? 0.5f * sizeX : -0.5f * sizeX,
                               (i & 2) ? 0.5f * sizeY : -0.5f * sizeY,
                               (i & 4) ? 0.5f * sizeZ : -0.5f * sizeZ);
    }

    const int quads[6][4] = { {0, 2, 6, 4}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 3, 7, 6}, {0, 1, 3, 2}, {4, 5, 7, 6} };
    Triangle triangles[12];
    int materialIndices[12] = {};
    for (auto i = 0; i < 6; ++i)
    {
        triangles[2 * i + 0] = Triangle{ {quads[i][0], quads[i][1], quads[i][2]} };
        triangles[2 * i + 1] = Triangle{ {quads[i][0], quads[i][2], quads[i][3]} };
    }

    scene->addStaticMesh(scene->createStaticMesh(8, 12, 1, vertices, triangles, materialIndices, &material));
    scene->commit();

    auto simulator = ReflectionSimulatorFactory::create(sceneType, numRays, 1024, duration, 0, 1, 1, 1, 64, nullptr);

    CoordinateSpace3f source(Vector3f(0.23f * sizeX, 0.17f * sizeY, 0.12f * sizeZ));
    CoordinateSpace3f listener(Vector3f(-0.2f * sizeX, -0.13f * sizeY, -0.08f * sizeZ));
    Directivity directivity;

    EnergyField energyField(duration, 0);
    EnergyField* energyFields[] = { &energyField };

    JobGraph jobGraph;
    simulator->simulate(*scene, 1, &source, 1, &listener, &directivity, numRays, numBounces, duration, 0, 1.0f,
                        energyFields, jobGraph);

    std::atomic<bool> cancel(false);
    while (jobGraph.processNextJob(0, cancel)) {}

    Reverb reverb;
    ReverbEstimator::estimate(energyField, AirAbsorptionModel(std::vector<float>(Bands::kNumBands, 0.0f).data(), nullptr, nullptr), reverb);
    return reverb.reverbTimes[1];
}

float eyringReverbTime(float sizeX,
                       float sizeY,
                       float sizeZ,
                       float absorption)
{
    auto volume = sizeX * sizeY * sizeZ;
    auto area = 2.0f * (sizeX * sizeY + sizeY * sizeZ + sizeX * sizeZ);
    return 0.161f * volume / (-area * logf(1.0f - absorption));
}


// A closed, diffuse-walled sphere of the given radius (a subdivided icosahedron). It has no corners, so the only
// thing that can make its reverberation time differ from Eyring's formula is how the simulator treats the paths.
void makeSphere(float radius,
                int subdivisions,
                std::vector<Vector3f>& vertices,
                std::vector<Triangle>& triangles)
{
    const auto t = (1.0f + sqrtf(5.0f)) / 2.0f;
    const float base[12][3] = { {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
                                {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1} };
    const int faces[20][3] = { {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4}, {11, 10, 2},
                               {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9}, {4, 9, 5},
                               {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1} };

    for (auto i = 0; i < 12; ++i)
    {
        vertices.push_back(Vector3f::unitVector(Vector3f(base[i][0], base[i][1], base[i][2])));
    }

    std::vector<Triangle> current;
    for (auto i = 0; i < 20; ++i)
    {
        current.push_back(Triangle{ {faces[i][0], faces[i][1], faces[i][2]} });
    }

    for (auto level = 0; level < subdivisions; ++level)
    {
        std::map<std::pair<int, int>, int> midpoints;
        auto midpoint = [&](int a, int b)
        {
            auto key = std::make_pair(std::min(a, b), std::max(a, b));
            auto it = midpoints.find(key);
            if (it != midpoints.end())
                return it->second;
            vertices.push_back(Vector3f::unitVector(vertices[a] + vertices[b]));
            return midpoints[key] = static_cast<int>(vertices.size()) - 1;
        };

        std::vector<Triangle> next;
        for (const auto& tri : current)
        {
            auto a = tri.indices[0], b = tri.indices[1], c = tri.indices[2];
            auto ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
            next.push_back(Triangle{ {a, ab, ca} });
            next.push_back(Triangle{ {b, bc, ab} });
            next.push_back(Triangle{ {c, ca, bc} });
            next.push_back(Triangle{ {ab, bc, ca} });
        }
        current = next;
    }

    for (auto& vertex : vertices)
    {
        vertex = vertex * radius;
    }
    triangles = current;
}

// Mid-band reverberation time that Steam Audio's own estimator reads from the energy field of a source and a
// listener inside a sphere.
float simulatedSphereReverbTime(SceneType sceneType,
                          float radius,
                          float absorption,
                          int numRays,
                          int numBounces,
                          float duration)
{
    auto embree = (sceneType == SceneType::Embree) ? make_shared<EmbreeDevice>() : nullptr;
    auto scene = shared_ptr<IScene>(SceneFactory::create(sceneType, nullptr, nullptr, nullptr, nullptr, nullptr, embree, nullptr));

    Material material;
    for (auto i = 0; i < Bands::kNumBands; ++i)
    {
        material.absorption[i] = absorption;
        material.transmission[i] = 0.0f;
    }
    material.scattering = 1.0f;

    std::vector<Vector3f> vertices;
    std::vector<Triangle> triangles;
    makeSphere(radius, 3, vertices, triangles);
    std::vector<int> materialIndices(triangles.size(), 0);

    scene->addStaticMesh(scene->createStaticMesh(static_cast<int>(vertices.size()), static_cast<int>(triangles.size()), 1,
                                                 vertices.data(), triangles.data(), materialIndices.data(), &material));
    scene->commit();

    auto simulator = ReflectionSimulatorFactory::create(sceneType, numRays, 1024, duration, 0, 1, 1, 1, 64, nullptr);

    CoordinateSpace3f source(Vector3f(0.4f * radius, 0.2f * radius, 0.1f * radius));
    CoordinateSpace3f listener(Vector3f(-0.3f * radius, -0.2f * radius, -0.1f * radius));
    Directivity directivity;

    EnergyField energyField(duration, 0);
    EnergyField* energyFields[] = { &energyField };

    JobGraph jobGraph;
    simulator->simulate(*scene, 1, &source, 1, &listener, &directivity, numRays, numBounces, duration, 0, 0.25f,
                        energyFields, jobGraph);

    std::atomic<bool> cancel(false);
    while (jobGraph.processNextJob(0, cancel)) {}

    Reverb reverb;
    ReverbEstimator::estimate(energyField, AirAbsorptionModel(std::vector<float>(Bands::kNumBands, 0.0f).data(), nullptr, nullptr), reverb);
    return reverb.reverbTimes[1];
}

// 4V / S is the mean free path of a closed convex room; Eyring's formula follows from the energy lost per bounce.
float eyringSphereReverbTime(float radius,
                       float absorption)
{
    const auto pi = 3.14159265f;
    auto volume = (4.0f / 3.0f) * pi * radius * radius * radius;
    auto area = 4.0f * pi * radius * radius;
    return 0.161f * volume / (-area * logf(1.0f - absorption));
}


const float kBoxSize[3] = { 3.0f, 3.0f, 2.5f };
const float kSourcePosition[3] = { 0.7f, 0.5f, 0.3f };
const float kListenerPosition[3] = { -0.6f, -0.4f, -0.2f };

// Energy that reaches the listener through specular reflections that bounce off the walls of a box (centered at the
// origin) exactly once or twice, as given by the image source method: each image of the source gives
// (1 - absorption)^(number of reflections) / (4 pi d^2), where d is the distance from the image to the listener.
float imageSourceEnergy(float absorption)
{
    const auto pi = 3.14159265f;

    auto total = 0.0f;
    for (auto nx = -2; nx <= 2; ++nx) for (auto px = 0; px <= 1; ++px)
    for (auto ny = -2; ny <= 2; ++ny) for (auto py = 0; py <= 1; ++py)
    for (auto nz = -2; nz <= 2; ++nz) for (auto pz = 0; pz <= 1; ++pz)
    {
        const int n[3] = { nx, ny, nz };
        const int p[3] = { px, py, pz };

        auto numReflections = 0;
        auto distanceSquared = 0.0f;
        for (auto i = 0; i < 3; ++i)
        {
            numReflections += abs(2 * n[i] - p[i]);

            // Wall positions are at +/- size / 2, so the images are laid out relative to a corner of the box.
            auto source = kSourcePosition[i] + 0.5f * kBoxSize[i];
            auto listener = kListenerPosition[i] + 0.5f * kBoxSize[i];
            auto image = 2.0f * n[i] * kBoxSize[i] + (p[i] ? -source : source);
            distanceSquared += (image - listener) * (image - listener);
        }

        if (numReflections < 1 || numReflections > 2)
            continue;

        total += powf(1.0f - absorption, static_cast<float>(numReflections)) / (4.0f * pi * distanceSquared);
    }
    return total;
}

// Simulates two bounces in a box whose walls are perfect (but absorbing) mirrors, and returns the total energy
// (omnidirectional, mid-band) of the resulting energy field.
float simulatedEnergy(SceneType sceneType,
                      float absorption,
                      int numRays)
{
    const auto pi = 3.14159265f;
    const auto duration = 0.2f;

    auto embree = (sceneType == SceneType::Embree) ? make_shared<EmbreeDevice>() : nullptr;
    auto scene = shared_ptr<IScene>(SceneFactory::create(sceneType, nullptr, nullptr, nullptr, nullptr, nullptr, embree, nullptr));

    Material material;
    for (auto i = 0; i < Bands::kNumBands; ++i)
    {
        material.absorption[i] = absorption;
        material.transmission[i] = 0.0f;
    }
    material.scattering = 0.0f;

    Vector3f vertices[8];
    for (auto i = 0; i < 8; ++i)
    {
        vertices[i] = Vector3f((i & 1) ? 0.5f * kBoxSize[0] : -0.5f * kBoxSize[0],
                               (i & 2) ? 0.5f * kBoxSize[1] : -0.5f * kBoxSize[1],
                               (i & 4) ? 0.5f * kBoxSize[2] : -0.5f * kBoxSize[2]);
    }

    const int quads[6][4] = { {0, 2, 6, 4}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 3, 7, 6}, {0, 1, 3, 2}, {4, 5, 7, 6} };
    Triangle triangles[12];
    int materialIndices[12] = {};
    for (auto i = 0; i < 6; ++i)
    {
        triangles[2 * i + 0] = Triangle{ {quads[i][0], quads[i][1], quads[i][2]} };
        triangles[2 * i + 1] = Triangle{ {quads[i][0], quads[i][2], quads[i][3]} };
    }

    scene->addStaticMesh(scene->createStaticMesh(8, 12, 1, vertices, triangles, materialIndices, &material));
    scene->commit();

    auto simulator = ReflectionSimulatorFactory::create(sceneType, numRays, 1024, duration, 0, 1, 1, 1, 64, nullptr);

    CoordinateSpace3f source(Vector3f(kSourcePosition[0], kSourcePosition[1], kSourcePosition[2]));
    CoordinateSpace3f listener(Vector3f(kListenerPosition[0], kListenerPosition[1], kListenerPosition[2]));
    Directivity directivity;

    EnergyField energyField(duration, 0);
    EnergyField* energyFields[] = { &energyField };

    JobGraph jobGraph;
    simulator->simulate(*scene, 1, &source, 1, &listener, &directivity, numRays, 2, duration, 0, 0.01f, energyFields,
                        jobGraph);

    std::atomic<bool> cancel(false);
    while (jobGraph.processNextJob(0, cancel)) {}

    auto total = 0.0f;
    for (auto i = 0; i < energyField.numBins(); ++i)
    {
        total += energyField[0][1][i];
    }

    // The order-0 spherical harmonic is 1 / (2 sqrt(pi)).
    return total * 2.0f * sqrtf(pi);
}

}

TEST_CASE("ReflectionSimulator", "[ReflectionSimulator]")
{
}

TEST_CASE("DecoupledReflectionSimulator", "[DecoupledReflectionSimulator]")
{
}

TEST_CASE("Simulated reverb time in a closed box matches Eyring's formula (CPU)", "[ReflectionSimulatorBox]")
{
    auto expected = eyringReverbTime(3.0f, 3.0f, 2.5f, 0.05f);
    auto actual = simulatedReverbTime(SceneType::Default, 3.0f, 3.0f, 2.5f, 0.05f, 1.0f, 16384, 600, 2.0f);
    INFO("Eyring: " << expected << " s, simulated: " << actual << " s");
    REQUIRE(fabsf(actual - expected) < 0.12f * expected);
}

#if defined(IPL_USES_EMBREE) && (defined(IPL_CPU_X86) || defined(IPL_CPU_X64))

TEST_CASE("Simulated reverb time in a closed box matches Eyring's formula (Embree)", "[ReflectionSimulatorBox]")
{
    auto expected = eyringReverbTime(3.0f, 3.0f, 2.5f, 0.05f);
    auto actual = simulatedReverbTime(SceneType::Embree, 3.0f, 3.0f, 2.5f, 0.05f, 1.0f, 16384, 600, 2.0f);
    INFO("Eyring: " << expected << " s, simulated: " << actual << " s");
    REQUIRE(fabsf(actual - expected) < 0.12f * expected);
}

#endif

// A point source and a listener do not absorb sound, so their 10 cm spheres must not stop the paths that cross them.
TEST_CASE("Source and listener do not absorb sound in a closed sphere (CPU)", "[ReflectionSimulatorSphere]")
{
    auto expected = eyringSphereReverbTime(1.0f, 0.02f);
    auto actual = simulatedSphereReverbTime(SceneType::Default, 1.0f, 0.02f, 16384, 1200, 4.0f);
    INFO("Eyring: " << expected << " s, simulated: " << actual << " s");
    REQUIRE(fabsf(actual - expected) < 0.1f * expected);
}

#if defined(IPL_USES_EMBREE) && (defined(IPL_CPU_X86) || defined(IPL_CPU_X64))

TEST_CASE("Source and listener do not absorb sound in a closed sphere (Embree)", "[ReflectionSimulatorSphere]")
{
    auto expected = eyringSphereReverbTime(1.0f, 0.02f);
    auto actual = simulatedSphereReverbTime(SceneType::Embree, 1.0f, 0.02f, 16384, 1200, 4.0f);
    INFO("Eyring: " << expected << " s, simulated: " << actual << " s");
    REQUIRE(fabsf(actual - expected) < 0.1f * expected);
}

#endif

TEST_CASE("Specular reflections carry the energy of the image sources (CPU)", "[ReflectionSimulatorSpecular]")
{
    auto expected = imageSourceEnergy(0.05f);
    auto actual = simulatedEnergy(SceneType::Default, 0.05f, 65536);
    INFO("image sources: " << expected << ", simulated: " << actual);
    REQUIRE(fabsf(actual - expected) < 0.08f * expected);
}

#if defined(IPL_USES_EMBREE) && (defined(IPL_CPU_X86) || defined(IPL_CPU_X64))

TEST_CASE("Specular reflections carry the energy of the image sources (Embree)", "[ReflectionSimulatorSpecular]")
{
    auto expected = imageSourceEnergy(0.05f);
    auto actual = simulatedEnergy(SceneType::Embree, 0.05f, 65536);
    INFO("image sources: " << expected << ", simulated: " << actual);
    REQUIRE(fabsf(actual - expected) < 0.08f * expected);
}

#endif

#if defined(IPL_USES_EMBREE) && (defined(IPL_CPU_X86) || defined(IPL_CPU_X64))

namespace {

// Total mid-band energy of the energy field from two bounces in a box (3 x 3 x 2.5 m before any rotation) whose walls
// are perfect, absorbing mirrors. If the box is instanced, it is rotated by 90 degrees about the y axis, which makes
// it 2.5 x 3 x 3 m; otherwise it is built directly with those dimensions.
float simulatedInstancedEnergy(bool instanced)
{
    const auto numRays = 16384;
    const auto duration = 0.2f;

    auto embree = make_shared<EmbreeDevice>();
    auto createScene = [&]()
    {
        return shared_ptr<IScene>(SceneFactory::create(SceneType::Embree, nullptr, nullptr, nullptr, nullptr, nullptr, embree, nullptr));
    };

    Material material;
    for (auto i = 0; i < Bands::kNumBands; ++i)
    {
        material.absorption[i] = 0.05f;
        material.transmission[i] = 0.0f;
    }
    material.scattering = 0.0f;

    const float size[3] = { instanced ? 3.0f : 2.5f, 3.0f, instanced ? 2.5f : 3.0f };
    Vector3f vertices[8];
    for (auto i = 0; i < 8; ++i)
    {
        vertices[i] = Vector3f((i & 1) ? 0.5f * size[0] : -0.5f * size[0],
                               (i & 2) ? 0.5f * size[1] : -0.5f * size[1],
                               (i & 4) ? 0.5f * size[2] : -0.5f * size[2]);
    }

    const int quads[6][4] = { {0, 2, 6, 4}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 3, 7, 6}, {0, 1, 3, 2}, {4, 5, 7, 6} };
    Triangle triangles[12];
    int materialIndices[12] = {};
    for (auto i = 0; i < 6; ++i)
    {
        triangles[2 * i + 0] = Triangle{ {quads[i][0], quads[i][1], quads[i][2]} };
        triangles[2 * i + 1] = Triangle{ {quads[i][0], quads[i][2], quads[i][3]} };
    }

    auto scene = createScene();
    auto subScene = createScene();
    subScene->addStaticMesh(subScene->createStaticMesh(8, 12, 1, vertices, triangles, materialIndices, &material));
    subScene->commit();

    if (instanced)
    {
        // (x, y, z) -> (z, y, -x)
        auto rotation = Matrix4x4f::identityMatrix();
        rotation(0, 0) = 0.0f;
        rotation(0, 2) = 1.0f;
        rotation(2, 0) = -1.0f;
        rotation(2, 2) = 0.0f;
        scene->addInstancedMesh(scene->createInstancedMesh(subScene, rotation));
    }
    else
    {
        scene->addStaticMesh(scene->createStaticMesh(8, 12, 1, vertices, triangles, materialIndices, &material));
    }
    scene->commit();

    auto simulator = ReflectionSimulatorFactory::create(SceneType::Embree, numRays, 1024, duration, 0, 1, 1, 1, 64, nullptr);

    CoordinateSpace3f source(Vector3f(0.5f, 0.5f, 0.3f));
    CoordinateSpace3f listener(Vector3f(-0.6f, -0.4f, -0.2f));
    Directivity directivity;

    EnergyField energyField(duration, 0);
    EnergyField* energyFields[] = { &energyField };

    JobGraph jobGraph;
    simulator->simulate(*scene, 1, &source, 1, &listener, &directivity, numRays, 2, duration, 0, 0.01f, energyFields,
                        jobGraph);

    std::atomic<bool> cancel(false);
    while (jobGraph.processNextJob(0, cancel)) {}

    auto total = 0.0f;
    for (auto i = 0; i < energyField.numBins(); ++i)
    {
        total += energyField[0][1][i];
    }
    return total;
}

}

TEST_CASE("Reflections in a rotated instanced mesh match those in the same geometry built directly", "[ReflectionSimulatorInstanced]")
{
    auto expected = simulatedInstancedEnergy(false);
    auto actual = simulatedInstancedEnergy(true);
    INFO("direct: " << expected << ", instanced: " << actual);
    REQUIRE(expected > 0.0f);
    REQUIRE(fabsf(actual - expected) < 0.03f * expected);
}

#endif
