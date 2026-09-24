# Third-party notices

The code in this repository is released under the MIT License ([`LICENSE`](LICENSE)), except for
the portions listed here.

## BoofCV

- Project: BoofCV, <https://github.com/lessthanoptimal/BoofCV> (Copyright (c) 2021, Peter Abeles)
- License: Apache License 2.0, [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)
- Upstream version: 0.44, the version this project depends on (`build.gradle`)

Two files adapt BoofCV source:

| File in this repository | Adapted from (BoofCV 0.44, module `boofcv-sfm`) |
|---|---|
| `src/main/java/org/boofcv/stitching/StitchingFactory.java` | `boofcv/factory/sfm/FactoryMotion2D.java`: `createMotion2D` builds the motion-estimation stack, including its fixed RANSAC seed. `createInstrumentedMotion2D`, `createSimilarityMotion2D` and `createProbedMotion2D` repeat that construction sequence. |
| `src/main/java/org/boofcv/stitching/SimilarityStitchingTransform.java` | `boofcv/alg/sfm/d2/FactoryStitchingTransform.java`: `createAffine_F64`, adapted to a similarity transform |

Upstream sources:
[`FactoryMotion2D.java`](https://github.com/lessthanoptimal/BoofCV/blob/v0.44/main/boofcv-sfm/src/main/java/boofcv/factory/sfm/FactoryMotion2D.java),
[`FactoryStitchingTransform.java`](https://github.com/lessthanoptimal/BoofCV/blob/v0.44/main/boofcv-sfm/src/main/java/boofcv/alg/sfm/d2/FactoryStitchingTransform.java).
BoofCV has no NOTICE file.

BoofCV is also used as an ordinary Maven dependency (`org.boofcv:boofcv-core`). Other Java and
Python dependencies are downloaded at build or install time under their own licenses and are not
redistributed here.

## Gradle wrapper

`gradlew`, `gradlew.bat` and `gradle/wrapper/` come from the Gradle project, under the Apache
License 2.0.
