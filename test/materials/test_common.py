import numpy as np
import safe_autonomy_simulation


def test_BLACK():
    assert np.all(
        safe_autonomy_simulation.materials.BLACK.specular == np.array([0.0, 0.0, 0.0])
    )
    assert np.all(
        safe_autonomy_simulation.materials.BLACK.diffuse == np.array([0.0, 0.0, 0.0])
    )
    assert np.all(
        safe_autonomy_simulation.materials.BLACK.ambient == np.array([0.0, 0.0, 0.0])
    )
    assert safe_autonomy_simulation.materials.BLACK.shininess == 0.0
    assert safe_autonomy_simulation.materials.BLACK.reflection == 0.0


def test_LIGHT():
    assert np.all(
        safe_autonomy_simulation.materials.LIGHT.specular == np.array([1.0, 1.0, 1.0])
    )
    assert np.all(
        safe_autonomy_simulation.materials.LIGHT.diffuse == np.array([1.0, 1.0, 1.0])
    )
    assert np.all(
        safe_autonomy_simulation.materials.LIGHT.ambient == np.array([1.0, 1.0, 1.0])
    )
    assert safe_autonomy_simulation.materials.LIGHT.shininess == 0.0
    assert safe_autonomy_simulation.materials.LIGHT.reflection == 0.0


def test_METALLIC_GREY():
    assert np.all(
        safe_autonomy_simulation.materials.METALLIC_GREY.specular == np.array([1, 1, 1])
    )
    assert np.all(
        safe_autonomy_simulation.materials.METALLIC_GREY.diffuse
        == np.array([0.1, 0.1, 0.1])
    )
    assert np.all(
        safe_autonomy_simulation.materials.METALLIC_GREY.ambient
        == np.array([0.4, 0.4, 0.4])
    )
    assert safe_autonomy_simulation.materials.METALLIC_GREY.shininess == 100.0
    assert safe_autonomy_simulation.materials.METALLIC_GREY.reflection == 0.0
