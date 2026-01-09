## [2.2.1](https://github.com/act3-ace/safe-autonomy-simulation/compare/v2.2.0...v2.2.1) (2026-01-09)


### Bug Fixes

* **deps:** downgrade numpy dependency to 1.x to allow for use with rllib ([#46](https://github.com/act3-ace/safe-autonomy-simulation/issues/46)) ([72d0b5d](https://github.com/act3-ace/safe-autonomy-simulation/commit/72d0b5d7615ccf8ad5d975b898f5c1efc066d69f))

# [2.2.0](https://github.com/act3-ace/safe-autonomy-simulation/compare/v2.1.6...v2.2.0) (2025-08-21)


### Features

* **camera:** move inspection logic to camera and allow camera off state ([#37](https://github.com/act3-ace/safe-autonomy-simulation/issues/37)) ([751465d](https://github.com/act3-ace/safe-autonomy-simulation/commit/751465d2e1ff845c2ac7c8f3ed105bb13c93a6a0))

## [2.1.6](https://github.com/act3-ace/safe-autonomy-simulation/compare/v2.1.5...v2.1.6) (2025-07-08)


### Bug Fixes

* Fix inverse rotation ([#35](https://github.com/act3-ace/safe-autonomy-simulation/issues/35)) ([641132f](https://github.com/act3-ace/safe-autonomy-simulation/commit/641132f1ef28bfcf177aa2c1ba6bf5b603e1a92b))

## [2.1.5](https://github.com/act3-ace/safe-autonomy-simulation/compare/v2.1.4...v2.1.5) (2025-07-07)


### Bug Fixes

* Fix Angular Velocity Reference Frame ([#34](https://github.com/act3-ace/safe-autonomy-simulation/issues/34)) ([f33afda](https://github.com/act3-ace/safe-autonomy-simulation/commit/f33afdac54ded4e6832936d154f380e196e9bad1))

## [2.1.4](https://github.com/act3-ace/safe-autonomy-simulation/compare/v2.1.3...v2.1.4) (2025-01-09)


### Bug Fixes

* **inspection:** enable entity-specific inspection point attributionfixed agent inspection weight attribution bug and added single agent inspection points attribution ([#28](https://github.com/act3-ace/safe-autonomy-simulation/issues/28)) ([c3aa3a5](https://github.com/act3-ace/safe-autonomy-simulation/commit/c3aa3a5c7ba8cfcfc03de814c2fd8f3e35462c56))

## [2.1.3](https://github.com/act3-ace/safe-autonomy-simulation/compare/v2.1.2...v2.1.3) (2024-10-30)


### Bug Fixes

* allow targets to specify how inspection points should be weighted ([#20](https://github.com/act3-ace/safe-autonomy-simulation/issues/20)) ([1379717](https://github.com/act3-ace/safe-autonomy-simulation/commit/1379717cf74e4bbaf32e20c3e0c56616a05f0842))

## [2.1.2](https://github.com/act3-ace/safe-autonomy-simulation/compare/v2.1.1...v2.1.2) (2024-10-25)


### Bug Fixes

* use jax clip for jit compliance ([b83c860](https://github.com/act3-ace/safe-autonomy-simulation/commit/b83c860ef6285fc89dde04daacb3e1e8c6990eee)), closes [#18](https://github.com/act3-ace/safe-autonomy-simulation/issues/18)

## [2.1.1](https://github.com/act3-ace/safe-autonomy-simulation/compare/v2.1.0...v2.1.1) (2024-10-25)


### Bug Fixes

* **jax:** move jax support to optional experimental feature ([1b4a52d](https://github.com/act3-ace/safe-autonomy-simulation/commit/1b4a52dd84bb43d547cc1ac15e0d998413758808)), closes [#8](https://github.com/act3-ace/safe-autonomy-simulation/issues/8)

# [2.1.0](https://github.com/act3-ace/safe-autonomy-simulation/compare/v2.0.15...v2.1.0) (2024-10-24)


### Features

* add ability to weight only half the sphere of inspection points ([07cdbb6](https://github.com/act3-ace/safe-autonomy-simulation/commit/07cdbb6972bae333f629d811066b6adab307a49f))

## [2.0.15](https://github.com/act3-ace/safe-autonomy-simulation/compare/v2.0.14...v2.0.15) (2024-10-23)


### Bug Fixes

* update 6 DOF state transition to keep position and velocity adjacent rather than separated by the quaternion derivative components ([b6607a1](https://github.com/act3-ace/safe-autonomy-simulation/commit/b6607a1639771a7a11961fa4483516e9cdf1dc35))
* update sun entity to just maintain sun angle as the only state variable internally ([c9aa8c9](https://github.com/act3-ace/safe-autonomy-simulation/commit/c9aa8c992146ba435ccdb4d318290a8c11139b35))

## [2.0.14](https://github.com/act3-ace/safe-autonomy-simulation/compare/v2.0.13...v2.0.14) (2024-08-30)


### Bug Fixes

* [#6](https://github.com/act3-ace/safe-autonomy-simulation/issues/6) passing control limits on entity creation ([2cf45c7](https://github.com/act3-ace/safe-autonomy-simulation/commit/2cf45c7b9e08595a1cfccadf827188dbcea61827))

## [2.0.13](https://github.com/act3-ace/safe-autonomy-simulation/compare/v2.0.12...v2.0.13) (2024-08-20)


### Bug Fixes

* [#3](https://github.com/act3-ace/safe-autonomy-simulation/issues/3) entity name property ([314bb7e](https://github.com/act3-ace/safe-autonomy-simulation/commit/314bb7eee090c8b41ffab7cca356a192ce5c9b2e))

## [2.0.12](https://github.com/act3-ace/safe-autonomy-simulation/compare/v2.0.11...v2.0.12) (2024-07-30)


### Bug Fixes

* **inspection point set:** initialize inspection points with normalized weights ([3a26587](https://github.com/act3-ace/safe-autonomy-simulation/commit/3a2658734727c2d43204c4c4977766d3b4778e54)), closes [#38](https://github.com/act3-ace/safe-autonomy-simulation/issues/38)

## [2.0.11]() (2024-07-17)


### Bug Fixes

* **inspection points:** ensure `InspectionPointSet` state is properly set during `step()` ([953afdf]()), closes [#35]()

## [2.0.10]() (2024-07-17)


### Bug Fixes

* **camera:** assume camera points at origin when not on sixdof spacecraft ([b3a2b77]()), closes [#33]()

## [2.0.9]() (2024-07-12)


### Bug Fixes

* **camera:** set the camera state in the inspector `reset()` method ([2cc4e6d]()), closes [#32]()

## [2.0.8]() (2024-07-12)


### Bug Fixes

* **inspection:** set kmeans n_init to 1 to suppress runtime warning ([8a37a4f]()), closes [#31]()

## [2.0.7]() (2024-07-12)


### Bug Fixes

* **sixdof:** use valid default bound types for sixdof state and state dot ([62b0d3f]()), closes [#29]()

## [2.0.6]() (2024-07-12)


### Bug Fixes

* **controls:** unambiguous truth value check for control bounds ([76e2fc1]()), closes [#28]()

## [2.0.5]() (2024-07-09)


### Bug Fixes

* **entity:** remove angle wrapping on physical entity angular velocity ([9901ca1]()), closes [#26]()

## [2.0.4]() (2024-07-09)


### Bug Fixes

* **inspection:** properly update inspection sim entities on step ([6baa1af]())

## [2.0.3]() (2024-07-01)


### Bug Fixes

* **ode:** use correct positional arg order for jax solver ([d1f5878]()), closes [#25]()

## [2.0.2]() (2024-06-28)


### Bug Fixes

* **deps:** use typing-extensions ([502ad98]())

## [2.0.1]() (2024-06-28)


### Bug Fixes

* **deps:** allow backward compatibility with older python versions ([2f4ed6d]()), closes [#23]()

# [2.0.0]() (2024-06-27)


* refactor(api)!: generalize api for building continuous simulations ([a0c7066]()), closes [#20]()


### BREAKING CHANGES

* api refactor such that simulators now constructed from a set of managed entity objects

Merge branch '20-add-type-hinting' into 'main'

Resolve "Add Type Hinting"

# [1.3.0]() (2024-04-15)


### Features

* **jax:** add jax support and rotational spacecraft model ([52e0f4b]()), closes [#19]()

# [1.2.0]() (2024-04-15)


### Features

* add point mass integrators ([65a1d9f]()), closes [#18]()

# [1.1.0]() (2024-04-05)


### Bug Fixes

* **gymnasium:** fix invalid private method args ([d98ec14]()), closes [#16]()
* rendered project blueprints (checkpoint commit made by act3-pt) ([dcba879]())


### Features

* **gym:** add example gymnasiums environments using docking and inspection simulators ([5a8abf3]()), closes [#14]()

## [1.0.1]() (2023-09-19)


### Bug Fixes

* **deps:** update dependency devsecops/cicd/pipeline to v14.0.16 ([4e17f49]())

# 1.0.0 (2023-09-19)


### Bug Fixes

* **deps:** update project with act3-pt (checkpoint commit made by act3-pt) (python-library:v1.0.13, branched from 1c9f33b) ([43fc57c]())
* **release:** fix dockerfile syntax ([6aead37]())
* **release:** make release.sh executable ([1e9a567]()), closes [#10]()
* **release:** remove unused poetry dep groups ([eb89fc5]())
* **release:** set release.sh as executable ([9800c5a]())
* **release:** set workdir in ci stage ([c55a4cd]())
* **test:** test release process ([0d28e77]())
* update dependencies and trigger initial release ([6550718]()), closes [#9]()


### Features

* add docking simulator (partially tested) ([c378504]()), closes [#1]()
