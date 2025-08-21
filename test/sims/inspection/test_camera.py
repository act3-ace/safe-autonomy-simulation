import pytest
import numpy as np
import safe_autonomy_simulation
import scipy.spatial.transform as transform


@pytest.mark.parametrize(
    "name, fov, resolution, focal_length, pixel_pitch",
    [
        ("camera", np.pi / 2, (1920, 1080), 1, 1e-3),
    ],
)
def test_init_defaults(name, fov, resolution, focal_length, pixel_pitch):
    camera = safe_autonomy_simulation.sims.inspection.Camera(
        name=name,
        fov=fov,
        resolution=resolution,
        focal_length=focal_length,
        pixel_pitch=pixel_pitch,
    )
    assert camera.name == name
    assert camera.fov == fov
    assert camera.resolution == resolution
    assert camera.focal_length == focal_length
    assert camera.pixel_pitch == pixel_pitch
    assert np.all(camera.position == np.zeros(3))
    assert np.all(camera.velocity == np.zeros(3))
    assert np.all(camera.orientation == np.array([0, 0, 0, 1]))
    assert np.all(camera.angular_velocity == np.zeros(3))
    assert isinstance(camera.control_queue, safe_autonomy_simulation.controls.NoControl)
    assert camera.material == safe_autonomy_simulation.materials.BLACK
    assert camera.parent is None
    assert len(camera.children) == 0


@pytest.mark.parametrize(
    "name, fov, resolution, focal_length, pixel_pitch, position, velocity, orientation, angular_velocity, control_queue, material, parent, children",
    [
        (
            "camera",
            np.pi / 2,
            [1920, 1080],
            1,
            1e-3,
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([0, 0, 0, 1]),
            np.array([7, 8, 9]),
            safe_autonomy_simulation.controls.NoControl(),
            safe_autonomy_simulation.materials.BLACK,
            safe_autonomy_simulation.entities.Point(
                "parent", position=np.array([10, 11, 12])
            ),
            [
                safe_autonomy_simulation.entities.Point(
                    "child", position=np.array([13, 14, 15])
                )
            ],
        ),
    ],
)
def test_init(
    name,
    fov,
    resolution,
    focal_length,
    pixel_pitch,
    position,
    velocity,
    orientation,
    angular_velocity,
    control_queue,
    material,
    parent,
    children,
):
    camera = safe_autonomy_simulation.sims.inspection.Camera(
        name=name,
        fov=fov,
        resolution=resolution,
        focal_length=focal_length,
        pixel_pitch=pixel_pitch,
        position=position,
        velocity=velocity,
        orientation=orientation,
        angular_velocity=angular_velocity,
        control_queue=control_queue,
        material=material,
        parent=parent,
        children=children,
    )
    assert camera.name == name
    assert camera.fov == fov
    assert camera.resolution == resolution
    assert camera.focal_length == focal_length
    assert camera.pixel_pitch == pixel_pitch
    assert np.all(camera.position == position)
    assert np.all(camera.velocity == velocity)
    assert np.all(camera.orientation == orientation)
    assert np.all(camera.angular_velocity == angular_velocity)
    assert camera.control_queue == control_queue
    assert camera.material == material
    assert camera.parent == parent
    assert len(camera.children) == len(children)
    for child in children:
        assert child in camera.children


# @pytest.mark.parametrize(
#     "camera, target",
#     [
#         (
#             safe_autonomy_simulation.sims.inspection.Camera(
#                 name="camera",
#                 fov=np.pi / 2,
#                 resolution=[1920, 1080],
#                 focal_length=1,
#                 pixel_pitch=1e-3,
#                 position=np.array([0, 0, 0]),
#                 velocity=np.array([0, 0, 0]),
#                 orientation=np.array([0, 0, 0, 1]),
#                 angular_velocity=np.array([0, 0, 0]),
#             ),
#             safe_autonomy_simulation.entities.Point(
#                 name="target", position=np.array([1, 0, 0])
#             ),
#         ),
#     ],
# )
# def test_point_at(camera, target):
#     direction = (target.position - camera.position) / np.linalg.norm(
#         target.position - camera.position
#     )
#     camera.point_at(target)
#     assert np.all(
#         camera.orientation == transform.Rotation.from_euler("xyz", direction).as_quat()
#     )

@pytest.mark.parametrize(
    "camera, point, light, viewed_object, radius, binary_ray, expected",
    [
        (
            safe_autonomy_simulation.sims.inspection.Camera(
                name="camera",
                fov=np.pi / 2,
                resolution=[1920, 1080],
                focal_length=1,
                pixel_pitch=1e-3,
                position=np.array([0, 0, 0]),
                velocity=np.array([0, 0, 0]),
                orientation=np.array([0, 0, 0, 1]),
                angular_velocity=np.array([0, 0, 0]),
                parent=None,
                children=[],
            ),
            safe_autonomy_simulation.entities.Point(
                name="point", position=np.array([1, 0, 0])
            ),
            safe_autonomy_simulation.entities.Point(
                name="light",
                position=np.array([0, 0, 1]),
                material=safe_autonomy_simulation.materials.LIGHT,
            ),
            safe_autonomy_simulation.entities.Point(
                name="viewed_object", position=np.array([1, 0, 0])
            ),
            1,
            False,
            False,
        ),
        (
            safe_autonomy_simulation.sims.inspection.Camera(
                name="camera",
                fov=np.pi / 2,
                resolution=[1920, 1080],
                focal_length=1,
                pixel_pitch=1e-3,
                position=np.array([0, 0, 0]),
                velocity=np.array([0, 0, 0]),
                orientation=np.array([0, 0, 0, 1]),
                angular_velocity=np.array([0, 0, 0]),
                parent=None,
                children=[],
            ),
            safe_autonomy_simulation.entities.Point(
                name="point", position=np.array([1, 0, 0])
            ),
            safe_autonomy_simulation.entities.Point(
                name="light",
                position=np.array([0, 0, 1]),
                material=safe_autonomy_simulation.materials.LIGHT,
            ),
            safe_autonomy_simulation.entities.Point(
                name="viewed_object", position=np.array([1, 0, 0])
            ),
            1,
            True,
            False,
        ),
    ],
)
def test_inspect_point(camera, point, light, viewed_object, radius, binary_ray, expected):
    result = camera.inspect_point(point=point, light=light, viewed_object=viewed_object, radius=radius, binary_ray=binary_ray)
    assert isinstance(result, bool)
    assert result == expected


@pytest.mark.parametrize(
    "camera, point, light, viewed_object, radius, binary_ray, expected",
    [
        (
            safe_autonomy_simulation.sims.inspection.Camera(
                name="camera",
                fov=np.pi / 2,
                resolution=[1920, 1080],
                focal_length=1,
                pixel_pitch=1e-3,
                position=np.array([0, 0, 0]),
                velocity=np.array([0, 0, 0]),
                orientation=np.array([0, 0, 0, 1]),
                angular_velocity=np.array([0, 0, 0]),
                parent=None,
                children=[],
            ),
            safe_autonomy_simulation.entities.Point(
                name="point", position=np.array([1, 0, 0])
            ),
            safe_autonomy_simulation.entities.Point(
                name="light",
                position=np.array([0, 0, 1]),
                material=safe_autonomy_simulation.materials.LIGHT,
            ),
            safe_autonomy_simulation.entities.Point(
                name="viewed_object", position=np.array([1, 0, 0])
            ),
            1,
            False,
            False,
        ),
        (
            safe_autonomy_simulation.sims.inspection.Camera(
                name="camera",
                fov=np.pi / 2,
                resolution=[1920, 1080],
                focal_length=1,
                pixel_pitch=1e-3,
                position=np.array([0, 0, 0]),
                velocity=np.array([0, 0, 0]),
                orientation=np.array([0, 0, 0, 1]),
                angular_velocity=np.array([0, 0, 0]),
                parent=None,
                children=[],
            ),
            safe_autonomy_simulation.entities.Point(
                name="point", position=np.array([1, 0, 0])
            ),
            safe_autonomy_simulation.entities.Point(
                name="light",
                position=np.array([0, 0, 1]),
                material=safe_autonomy_simulation.materials.LIGHT,
            ),
            safe_autonomy_simulation.entities.Point(
                name="viewed_object", position=np.array([1, 0, 0])
            ),
            1,
            True,
            False,
        ),
    ],
)
def test_check_point_illumination(
    camera, point, light, viewed_object, radius, binary_ray, expected
):
    assert (
        camera.check_point_illumination(point, light, viewed_object, radius, binary_ray)
        == expected
    )


@pytest.mark.parametrize(
    "camera, point, light, viewed_object, radius, expected",
    [
        (
            safe_autonomy_simulation.sims.inspection.Camera(
                name="camera",
                fov=np.pi / 2,
                resolution=[1920, 1080],
                focal_length=1,
                pixel_pitch=1e-3,
                position=np.array([0, 0, 0]),
                velocity=np.array([0, 0, 0]),
                orientation=np.array([0, 0, 0, 1]),
                angular_velocity=np.array([0, 0, 0]),
                parent=None,
                children=[],
            ),
            safe_autonomy_simulation.entities.Point(
                name="point", position=np.array([1, 0, 0])
            ),
            safe_autonomy_simulation.entities.Point(
                name="light",
                position=np.array([0, 0, 1]),
                material=safe_autonomy_simulation.materials.LIGHT,
            ),
            safe_autonomy_simulation.entities.Point(
                name="viewed_object", position=np.array([1, 0, 0])
            ),
            1,
            np.array([0, 0, 0]),
        ),
    ],
)
def test_capture_point(camera, point, light, viewed_object, radius, expected):
    assert np.all(camera.capture_point(point, light, viewed_object, radius) == expected)


@pytest.mark.parametrize(
    "camera, light, viewed_object, radius, expected",
    [
        (
            safe_autonomy_simulation.sims.inspection.Camera(
                name="camera",
                fov=np.pi / 2,
                resolution=[10, 5],
                focal_length=1,
                pixel_pitch=1e-3,
                position=np.array([0, 0, 0]),
                velocity=np.array([0, 0, 0]),
                orientation=np.array([0, 0, 0, 1]),
                angular_velocity=np.array([0, 0, 0]),
                parent=None,
                children=[],
            ),
            safe_autonomy_simulation.entities.Point(
                name="light",
                position=np.array([0, 0, 1]),
                material=safe_autonomy_simulation.materials.LIGHT,
            ),
            safe_autonomy_simulation.entities.Point(
                name="viewed_object", position=np.array([1, 0, 0])
            ),
            1,
            np.zeros((5, 10, 3)),
        ),
    ],
)
def test_capture(camera, light, viewed_object, radius, expected):
    assert np.all(camera.capture(light, viewed_object, radius) == expected)
