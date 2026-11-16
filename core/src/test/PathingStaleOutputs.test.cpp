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

#include <baked_reflection_data.h>
#include <path_visibility.h>
#include <sampling.h>
#include <scene_factory.h>
#include <simulation_data.h>
#include <simulation_manager.h>

using namespace ipl;


namespace
{

// Adds an axis-aligned quad in a plane of constant z, spanning [x0, x1] x [y0, y1], to the vertex and triangle lists.
void addQuadZ(vector<Vector3f>& vertices, vector<Triangle>& triangles, float x0, float x1, float y0, float y1, float z)
{
    auto base = static_cast<int>(vertices.size());
    vertices.push_back(Vector3f(x0, y0, z));
    vertices.push_back(Vector3f(x1, y0, z));
    vertices.push_back(Vector3f(x1, y1, z));
    vertices.push_back(Vector3f(x0, y1, z));

    Triangle a;
    a.indices[0] = base + 0; a.indices[1] = base + 1; a.indices[2] = base + 2;
    Triangle b;
    b.indices[0] = base + 0; b.indices[1] = base + 2; b.indices[2] = base + 3;
    triangles.push_back(a);
    triangles.push_back(b);
}

shared_ptr<IScene> makeScene(const vector<Vector3f>& vertices, const vector<Triangle>& triangles)
{
    shared_ptr<IScene> scene(SceneFactory::create(SceneType::Default, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));

    if (!triangles.empty())
    {
        vector<int> materialIndices(triangles.size(), 0);
        Material materials[1] = {};
        scene->addStaticMesh(scene->createStaticMesh(static_cast<int>(vertices.size()), static_cast<int>(triangles.size()), 1,
                                                     vertices.data(), triangles.data(), materialIndices.data(), materials));
    }

    scene->commit();
    return scene;
}

}

namespace
{

struct PathingRig
{
    shared_ptr<IScene> scene;
    unique_ptr<SimulationManager> manager;
    shared_ptr<ProbeBatch> probes;
    shared_ptr<SimulationData> source;

    PathingRig(shared_ptr<IScene> scene, const Sphere& probe)
        : scene(scene)
    {
        manager = make_unique<SimulationManager>(false, false, true, SceneType::Default, IndirectEffectType::Parametric, 32, 1024, 32,
                                                 1.0f, 1, 4, 1, 1, 1, 8, true, Vector3f(0.0f, -1.0f, 0.0f), 48000, 1024,
                                                 nullptr, nullptr, nullptr);
        manager->scene() = scene;

        probes = make_shared<ProbeBatch>();
        probes->addProbe(probe);
        probes->commit();

        manager->addProbeBatch(probes);

        source = make_shared<SimulationData>(false, true, SceneType::Default, IndirectEffectType::Parametric, 32, 1.0f, 1, 48000, 1024, nullptr, nullptr);
        source->pathingInputs.enabled = true;
        source->pathingInputs.probes = probes;
        source->pathingInputs.visRadius = 0.0f;
        source->pathingInputs.visThreshold = 0.1f;
        source->pathingInputs.visRange = 100.0f;
        source->pathingInputs.order = 1;
        source->pathingInputs.enableValidation = false;
        source->pathingInputs.findAlternatePaths = false;
        source->pathingInputs.simplifyPaths = true;
        source->pathingInputs.realTimeVis = true;
        manager->addSource(source);

        manager->commit();
    }

    void run(const Vector3f& listener, const Vector3f& sourcePosition)
    {
        SharedPathingSimulationInputs shared;
        shared.listener = CoordinateSpace3f(listener);
        manager->setSharedPathingInputs(shared);

        source->pathingInputs.source = CoordinateSpace3f(sourcePosition);
        manager->simulatePathing();
    }

    float shEnergy() const
    {
        auto energy = 0.0f;
        for (auto i = 0u; i < source->pathingOutputs.sh.totalSize(); ++i)
            energy += source->pathingOutputs.sh[i] * source->pathingOutputs.sh[i];
        return energy;
    }
};

}

// When the source is occluded from the listener and is not covered by any probe, there is no path from it to the
// listener. The simulator must report silence, as it does when probes exist but no path is found. It used to leave
// the outputs of the previous simulation untouched, so the source kept being heard along its last line-of-sight path.
TEST_CASE("Pathing outputs are cleared when an occluded source has no probes", "[Pathing]")
{
    vector<Vector3f> vertices;
    vector<Triangle> triangles;
    addQuadZ(vertices, triangles, -50.0f, 50.0f, -50.0f, 50.0f, 0.0f);
    auto scene = makeScene(vertices, triangles);

    // The only probe is next to the listener, on the near side of the wall.
    PathingRig rig(scene, Sphere(Vector3f(0.0f, 1.6f, -3.0f), 2.0f));

    // Line of sight: a direct path, audible.
    rig.run(Vector3f(0.0f, 1.6f, -3.0f), Vector3f(0.0f, 1.6f, -6.0f));
    auto before = rig.shEnergy();
    printf("sh energy with line of sight: %f\n", before);
    REQUIRE(before > 0.0f);

    // The source goes behind the wall, far outside the probe: no line of sight, no probe, no path.
    rig.run(Vector3f(0.0f, 1.6f, -3.0f), Vector3f(0.0f, 1.6f, 6.0f));
    auto after = rig.shEnergy();
    printf("sh energy behind the wall, outside the probes: %f\n", after);
    REQUIRE(after == 0.0f);
}
