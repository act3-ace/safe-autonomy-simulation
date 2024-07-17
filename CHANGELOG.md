## [2.0.10](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/compare/v2.0.9...v2.0.10) (2024-07-17)


### Bug Fixes

* **camera:** assume camera points at origin when not on sixdof spacecraft ([b3a2b77](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/b3a2b778ca7135c5b0c04c83b7c645b36f93d237)), closes [#33](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/33)

## [2.0.9](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/compare/v2.0.8...v2.0.9) (2024-07-12)


### Bug Fixes

* **camera:** set the camera state in the inspector `reset()` method ([2cc4e6d](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/2cc4e6dfd7f537fab57158c86a516e90de5aad5f)), closes [#32](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/32)

## [2.0.8](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/compare/v2.0.7...v2.0.8) (2024-07-12)


### Bug Fixes

* **inspection:** set kmeans n_init to 1 to suppress runtime warning ([8a37a4f](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/8a37a4f01a03a027f7bdbb573c97dff44106d56f)), closes [#31](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/31)

## [2.0.7](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/compare/v2.0.6...v2.0.7) (2024-07-12)


### Bug Fixes

* **sixdof:** use valid default bound types for sixdof state and state dot ([62b0d3f](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/62b0d3f85506513e9d905e313f8cec011916db3c)), closes [#29](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/29)

## [2.0.6](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/compare/v2.0.5...v2.0.6) (2024-07-12)


### Bug Fixes

* **controls:** unambiguous truth value check for control bounds ([76e2fc1](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/76e2fc142929407bebd097cb93c1a93da446899e)), closes [#28](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/28)

## [2.0.5](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/compare/v2.0.4...v2.0.5) (2024-07-09)


### Bug Fixes

* **entity:** remove angle wrapping on physical entity angular velocity ([9901ca1](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/9901ca1491c8abc9c0400cf54b95a45be65fef19)), closes [#26](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/26)

## [2.0.4](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/compare/v2.0.3...v2.0.4) (2024-07-09)


### Bug Fixes

* **inspection:** properly update inspection sim entities on step ([6baa1af](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/6baa1af29e5edef7b722ae93bc86005d7c0b850f))

## [2.0.3](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/compare/v2.0.2...v2.0.3) (2024-07-01)


### Bug Fixes

* **ode:** use correct positional arg order for jax solver ([d1f5878](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/d1f587828074de25a8cab623fd470645471a14ec)), closes [#25](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/25)

## [2.0.2](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/compare/v2.0.1...v2.0.2) (2024-06-28)


### Bug Fixes

* **deps:** use typing-extensions ([502ad98](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/502ad98aa53efd15a52438c124f2c922d0e554ba))

## [2.0.1](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/compare/v2.0.0...v2.0.1) (2024-06-28)


### Bug Fixes

* **deps:** allow backward compatibility with older python versions ([2f4ed6d](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/2f4ed6d298df503a0c616e8634cff4fefb246663)), closes [#23](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/23)

# [2.0.0](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/compare/v1.3.0...v2.0.0) (2024-06-27)


* refactor(api)!: generalize api for building continuous simulations ([a0c7066](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/a0c70661d385b8eb09e4c0b24c44517f3bb129bf)), closes [#20](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/20)


### BREAKING CHANGES

* api refactor such that simulators now constructed from a set of managed entity objects

Merge branch '20-add-type-hinting' into 'main'

Resolve "Add Type Hinting"

# [1.3.0](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/compare/v1.2.0...v1.3.0) (2024-04-15)


### Features

* **jax:** add jax support and rotational spacecraft model ([52e0f4b](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/52e0f4bafe08aeeffe93cc5d3a93850521e63e72)), closes [#19](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/19)

# [1.2.0](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/compare/v1.1.0...v1.2.0) (2024-04-15)


### Features

* add point mass integrators ([65a1d9f](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/65a1d9f211a300a0d980a44c2efac45b8b60a8bb)), closes [#18](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/18)

# [1.1.0](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/compare/v1.0.1...v1.1.0) (2024-04-05)


### Bug Fixes

* **gymnasium:** fix invalid private method args ([d98ec14](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/d98ec143fe88557536e4eac75555fae960cba5a0)), closes [#16](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/16)
* rendered project blueprints (checkpoint commit made by act3-pt) ([dcba879](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/dcba8799ac9dc7b4d77134a64b53f407330e5eec))


### Features

* **gym:** add example gymnasiums environments using docking and inspection simulators ([5a8abf3](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/5a8abf3e4f0585a5301287dc74e6da27c0e8e2c3)), closes [#14](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/14)

## [1.0.1](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/compare/v1.0.0...v1.0.1) (2023-09-19)


### Bug Fixes

* **deps:** update dependency devsecops/cicd/pipeline to v14.0.16 ([4e17f49](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/4e17f49dad0d507e3abcdf3c4cff659c038f9a87))

# 1.0.0 (2023-09-19)


### Bug Fixes

* **deps:** update project with act3-pt (checkpoint commit made by act3-pt) (python-library:v1.0.13, branched from 1c9f33b) ([43fc57c](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/43fc57c242bba22a176e91ccf36f939303172c56))
* **release:** fix dockerfile syntax ([6aead37](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/6aead37f6f8ee3353b7fa6849e9ebff99891c698))
* **release:** make release.sh executable ([1e9a567](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/1e9a567831519b494e9cbdc4801b28cd2089d37b)), closes [#10](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/10)
* **release:** remove unused poetry dep groups ([eb89fc5](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/eb89fc53392f3a3def1cf1c5adfd30c762f7d439))
* **release:** set release.sh as executable ([9800c5a](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/9800c5a94382f9ac1c003b3acce565052eac98d1))
* **release:** set workdir in ci stage ([c55a4cd](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/c55a4cd3f0271dce839421440b90d9de37d20970))
* **test:** test release process ([0d28e77](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/0d28e773f12277eb7ef3577f1cdcad829a589ca9))
* update dependencies and trigger initial release ([6550718](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/65507180a31dce7303c9ebf110bedcba19390396)), closes [#9](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/9)


### Features

* add docking simulator (partially tested) ([c378504](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/commit/c3785043c3fc4038efda08eb92e1e0eac2719d84)), closes [#1](https://git.act3-ace.com/rta/safe-autonomy-stack/safe-autonomy-simulation/issues/1)
