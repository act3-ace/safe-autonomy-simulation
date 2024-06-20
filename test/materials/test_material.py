import pytest
import numpy as np
import safe_autonomy_simulation


@pytest.mark.parametrize(
    "specular, diffuse, ambient, shininess, reflection",
    [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, 0.0),
        ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 0.0, 0.0),
        ([1, 1, 1], [0.1, 0.1, 0.1], [0.4, 0.4, 0.4], 100.0, 0.0),
    ],
)
def test_material(specular, diffuse, ambient, shininess, reflection):
    material = safe_autonomy_simulation.materials.Material(
        specular=specular,
        diffuse=diffuse,
        ambient=ambient,
        shininess=shininess,
        reflection=reflection,
    )
    assert np.all(material.specular == specular)
    assert np.all(material.diffuse == diffuse)
    assert np.all(material.ambient == ambient)
    assert material.shininess == shininess
    assert material.reflection == reflection
