# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [1.0.1] - 2024-05-10

### :warning: Deprecation Warning

-   The MrVI implementation has been moved to `scvi-tools` (`from scvi.external import MRVI`),
    and this package will no longer be maintained.

### Added

-   Adds deprecation warning in the form of a FutureWarning on import.
-   Various fixes and UX improvements to the model kwargs, DE, and DA functions.

## [1.0.0b1] - 2024-02-21

### :warning: Breaking Changes

-   This release is a major rework of the mrvi model and is generally incompatible with
    the previous v0.x.x releases.

### Added

-   New attention-based encoder/decoder components.
-   Model-based multivariate differential expression procedure at single-cell resolution.
-   Model-based differential abundance procedure at single-cell resolution.
-   Mismatched latent dimension sizes.
-   Optional mixture of gaussians prior. Can also make use of cell-type labels.
-   JAX implementation.
