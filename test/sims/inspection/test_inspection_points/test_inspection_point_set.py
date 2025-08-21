import pytest
import numpy as np
import safe_autonomy_simulation
import scipy.spatial.transform as transform


@pytest.mark.parametrize(
    "name, parent, num_points, radius, priority_vector",
    [
        (
            "set",
            safe_autonomy_simulation.entities.Point(
                "parent", position=np.array([0, 0, 0])
            ),
            10,
            1,
            np.array([1, 0, 0]),
        ),
        (
            "set",
            safe_autonomy_simulation.entities.Point(
                "parent", position=np.array([0, 0, 0])
            ),
            10,
            1,
            np.array([0, 1, 0]),
        ),
        (
            "set",
            safe_autonomy_simulation.entities.Point(
                "parent", position=np.array([0, 0, 0])
            ),
            10,
            1,
            np.array([0, 0, 1]),
        ),
    ],
)
def test_init_default(name, parent, num_points, radius, priority_vector):
    point_set = safe_autonomy_simulation.sims.inspection.InspectionPointSet(
        name=name,
        parent=parent,
        num_points=num_points,
        radius=radius,
        priority_vector=priority_vector,
    )
    assert point_set.name == name
    assert point_set.parent == parent
    assert point_set.num_points == len(point_set.points)
    assert point_set.radius == radius
    assert np.all(point_set.priority_vector == priority_vector)
    assert isinstance(
        point_set.dynamics, safe_autonomy_simulation.dynamics.PassThroughDynamics
    )
    assert isinstance(
        point_set.control_queue, safe_autonomy_simulation.controls.NoControl
    )
    for _, point in point_set.points.items():
        assert np.allclose(np.linalg.norm(point.position), radius)
        assert not point.inspected
        assert point.inspector is None
        assert point.parent == point_set
        assert point.name == "point"


@pytest.mark.parametrize(
    "name, parent, num_points, radius, priority_vector, dynamics, control_queue, material",
    [
        (
            "set",
            safe_autonomy_simulation.entities.Point(
                "parent", position=np.array([0, 0, 0])
            ),
            10,
            1,
            np.array([1, 0, 0]),
            safe_autonomy_simulation.dynamics.PassThroughDynamics(),
            safe_autonomy_simulation.controls.NoControl(),
            safe_autonomy_simulation.materials.BLACK,
        ),
        (
            "set",
            safe_autonomy_simulation.entities.Point(
                "parent", position=np.array([0, 0, 0])
            ),
            10,
            1,
            np.array([0, 1, 0]),
            safe_autonomy_simulation.dynamics.PassThroughDynamics(),
            safe_autonomy_simulation.controls.NoControl(),
            safe_autonomy_simulation.materials.BLACK,
        ),
        (
            "set",
            safe_autonomy_simulation.entities.Point(
                "parent", position=np.array([0, 0, 0])
            ),
            10,
            1,
            np.array([0, 0, 1]),
            safe_autonomy_simulation.dynamics.PassThroughDynamics(),
            safe_autonomy_simulation.controls.NoControl(),
            safe_autonomy_simulation.materials.BLACK,
        ),
    ],
)
def test_init_args(
    name, parent, num_points, radius, priority_vector, dynamics, control_queue, material
):
    point_set = safe_autonomy_simulation.sims.inspection.InspectionPointSet(
        name=name,
        parent=parent,
        num_points=num_points,
        radius=radius,
        priority_vector=priority_vector,
        dynamics=dynamics,
        control_queue=control_queue,
        material=material,
    )
    assert point_set.name == name
    assert point_set.parent == parent
    assert point_set.num_points == len(point_set.points)
    assert point_set.radius == radius
    assert np.all(point_set.priority_vector == priority_vector)
    assert point_set.dynamics == dynamics
    assert point_set.control_queue == control_queue
    assert point_set.material == material
    for _, point in point_set.points.items():
        assert np.allclose(np.linalg.norm(point.position), radius)
        assert not point.inspected
        assert point.inspector is None
        assert point.parent == point_set
        assert point.name == "point"


def test_build_initial_state():
    point_set = safe_autonomy_simulation.sims.inspection.InspectionPointSet(
        name="set",
        parent=safe_autonomy_simulation.entities.Point(
            "parent", position=np.array([0, 0, 0])
        ),
        num_points=10,
        radius=1,
        priority_vector=np.array([1, 0, 0]),
    )
    expected_state = np.array([p.state for p in point_set.points.values()])
    assert np.all(point_set.build_initial_state() == expected_state)


def test_reset():
    point_set = safe_autonomy_simulation.sims.inspection.InspectionPointSet(
        name="set",
        parent=safe_autonomy_simulation.entities.Point(
            "parent", position=np.array([0, 0, 0])
        ),
        num_points=10,
        radius=1,
        priority_vector=np.array([1, 0, 0]),
    )
    initial_weights = {i: p.weight for i, p in point_set.points.items()}
    point_set.reset()
    for i, point in point_set.points.items():
        assert np.allclose(np.linalg.norm(point.position), 1)
        assert not point.inspected
        assert point.inspector is None
        assert point.parent == point_set
        assert point.name == "point"
        assert point.weight == initial_weights[i]



def test__pre_step():
    point_set = safe_autonomy_simulation.sims.inspection.InspectionPointSet(
        name="set",
        parent=safe_autonomy_simulation.entities.Point(
            "parent", position=np.array([0, 0, 0])
        ),
        num_points=10,
        radius=1,
        priority_vector=np.array([1, 0, 0]),
    )
    point_set._pre_step(step_size=1)
    expected_state = np.array([p.state for p in point_set.points.values()])
    assert np.all(point_set.state == expected_state)


def test__post_step():
    point_set = safe_autonomy_simulation.sims.inspection.InspectionPointSet(
        name="set",
        parent=safe_autonomy_simulation.entities.Point(
            "parent", position=np.array([0, 0, 0])
        ),
        num_points=10,
        radius=1,
        priority_vector=np.array([1, 0, 0]),
    )
    point_set._post_step(step_size=1)
    expected_state = np.array([p.state for p in point_set.points.values()])
    assert np.all(point_set.state == expected_state)


@pytest.mark.parametrize(
    "num_points, points_algorithm, radius, priority_vector",
    [
        (10, "fibonacci", 1, np.array([1, 0, 0])),
        (10, "fibonacci", 1, np.array([0, 1, 0])),
        (10, "fibonacci", 1, np.array([0, 0, 1])),
        (10, "cmu", 1, np.array([1, 0, 0])),
        (10, "cmu", 1, np.array([0, 1, 0])),
        (10, "cmu", 1, np.array([0, 0, 1])),
    ],
)
def test__generate_points(num_points, points_algorithm, radius, priority_vector):
    point_set = safe_autonomy_simulation.sims.inspection.InspectionPointSet(
        name="set",
        parent=safe_autonomy_simulation.entities.Point(
            "parent", position=np.array([0, 0, 0])
        ),
        num_points=num_points,
        radius=radius,
        priority_vector=priority_vector,
        points_algorithm=points_algorithm,
    )
    if points_algorithm == "fibonacci":
        expected_positions = safe_autonomy_simulation.sims.inspection.utils.sphere.points_on_sphere_fibonacci(
            num_points, radius
        )
    else:
        expected_positions = (
            safe_autonomy_simulation.sims.inspection.utils.sphere.points_on_sphere_cmu(
                num_points, radius
            )
        )
    total_weight = sum(
        [
            np.arccos(
                np.dot(-priority_vector, pos)
                / (np.linalg.norm(-priority_vector) * np.linalg.norm(pos))
            )
            / np.pi
            for pos in expected_positions
        ]
    )
    point_set._generate_points(num_points=num_points, points_algorithm=points_algorithm)
    for _, point in point_set.points.items():
        expected_weight = (
            np.arccos(
                np.dot(-priority_vector, point.position)
                / (np.linalg.norm(-priority_vector) * np.linalg.norm(point.position))
            )
            / np.pi
        ) / total_weight
        assert np.allclose(np.linalg.norm(point.position), 1)
        assert not point.inspected
        assert np.allclose(point.weight, expected_weight)
        assert point.inspector is None
        assert point.parent == point_set
        assert point.name == "point"


@pytest.mark.parametrize(
    "point_set, camera, sun, binary_ray",
    [
        (
            safe_autonomy_simulation.sims.inspection.InspectionPointSet(
                name="set",
                parent=safe_autonomy_simulation.entities.Point(
                    "parent", position=np.array([0, 0, 0])
                ),
                num_points=10,
                radius=1,
                priority_vector=np.array([1, 0, 0]),
            ),
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
            False,
        ),
    ],
)
def test_update_points_inspection_status(point_set, camera, sun, binary_ray):
    point_set.update_points_inspection_status(camera, sun, binary_ray)
    for _, point in point_set.points.items():
        expected_inspected = camera.inspect_point(
            point=point,
            light=sun,
            viewed_object=point_set.parent,
            radius=point_set.radius,
            binary_ray=binary_ray,
        )
        if expected_inspected:
            assert point.inspected
            assert point.inspector == camera
        else:
            assert not point.inspected
            assert point.inspector is None


@pytest.mark.parametrize(
    "camera, sun, binary_ray",
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
            False,
        ),
    ],
)
def test_kmeans_find_nearest_cluster(camera, sun, binary_ray):
    point_set = safe_autonomy_simulation.sims.inspection.InspectionPointSet(
        name="set",
        parent=safe_autonomy_simulation.entities.Point(
            "parent", position=np.array([0, 0, 0])
        ),
        num_points=10,
        radius=1,
        priority_vector=np.array([1, 0, 0]),
    )
    point_set.update_points_inspection_status(camera, sun, binary_ray)
    nearest_cluster = point_set.kmeans_find_nearest_cluster(camera, sun, binary_ray)
    if len([p for p in point_set.points.values() if p.inspected]) == 0:
        expected_nearest_cluster = np.array([0.0, 0.0, 0.0])
    else:
        expected_nearest_cluster = point_set.clusters[
            np.argmin([np.linalg.norm(c - camera.position) for c in point_set.clusters])
        ]
        expected_nearest_cluster = expected_nearest_cluster / np.linalg.norm(
            expected_nearest_cluster
        )
    assert np.allclose(nearest_cluster, expected_nearest_cluster)


@pytest.mark.parametrize(
    "camera, sun, binary_ray",
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
            False,
        ),
    ],
)
def test_get_num_points_inspected_inspector(camera, sun, binary_ray):
    point_set = safe_autonomy_simulation.sims.inspection.InspectionPointSet(
        name="set",
        parent=safe_autonomy_simulation.entities.Point(
            "parent", position=np.array([0, 0, 0])
        ),
        num_points=10,
        radius=1,
        priority_vector=np.array([1, 0, 0]),
    )
    point_set.update_points_inspection_status(camera, sun, binary_ray)
    num_points_inspected = point_set.get_num_points_inspected(inspector_entity=camera)
    expected_num_points_inspected = len(
        [p for p in point_set.points.values() if p.inspected and p.inspector == camera]
    )
    assert num_points_inspected == expected_num_points_inspected


@pytest.mark.parametrize(
    "camera, sun, binary_ray",
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
            False,
        ),
    ],
)
def test_get_num_points_inspected(camera, sun, binary_ray):
    point_set = safe_autonomy_simulation.sims.inspection.InspectionPointSet(
        name="set",
        parent=safe_autonomy_simulation.entities.Point(
            "parent", position=np.array([0, 0, 0])
        ),
        num_points=10,
        radius=1,
        priority_vector=np.array([1, 0, 0]),
    )
    point_set.update_points_inspection_status(camera, sun, binary_ray)
    num_points_inspected = point_set.get_num_points_inspected()
    expected_num_points_inspected = len(
        [p for p in point_set.points.values() if p.inspected]
    )
    assert num_points_inspected == expected_num_points_inspected


@pytest.mark.parametrize(
    "camera, sun, binary_ray",
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
            False,
        ),
    ],
)
def test_get_percentage_of_points_inspected_inspector(camera, sun, binary_ray):
    point_set = safe_autonomy_simulation.sims.inspection.InspectionPointSet(
        name="set",
        parent=safe_autonomy_simulation.entities.Point(
            "parent", position=np.array([0, 0, 0])
        ),
        num_points=10,
        radius=1,
        priority_vector=np.array([1, 0, 0]),
    )
    point_set.update_points_inspection_status(camera, sun, binary_ray)
    percentage_of_points_inspected = point_set.get_percentage_of_points_inspected(
        inspector_entity=camera
    )
    expected_percentage_of_points_inspected = point_set.get_num_points_inspected(
        inspector_entity=camera
    ) / len(point_set.points)
    assert percentage_of_points_inspected == expected_percentage_of_points_inspected


@pytest.mark.parametrize(
    "camera, sun, binary_ray",
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
            False,
        ),
    ],
)
def get_percentage_of_points_inspected(camera, sun, binary_ray):
    point_set = safe_autonomy_simulation.sims.inspection.InspectionPointSet(
        name="set",
        parent=safe_autonomy_simulation.entities.Point(
            "parent", position=np.array([0, 0, 0])
        ),
        num_points=10,
        radius=1,
        priority_vector=np.array([1, 0, 0]),
    )
    point_set.update_points_inspection_status(camera, sun, binary_ray)
    percentage_of_points_inspected = point_set.get_percentage_of_points_inspected()
    expected_percentage_of_points_inspected = (
        point_set.get_num_points_inspected() / len(point_set.points)
    )
    assert percentage_of_points_inspected == expected_percentage_of_points_inspected


@pytest.mark.parametrize(
    "camera, sun, binary_ray",
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
            False,
        ),
    ],
)
def get_total_weight_inspected_inspector(camera, sun, binary_ray):
    point_set = safe_autonomy_simulation.sims.inspection.InspectionPointSet(
        name="set",
        parent=safe_autonomy_simulation.entities.Point(
            "parent", position=np.array([0, 0, 0])
        ),
        num_points=10,
        radius=1,
        priority_vector=np.array([1, 0, 0]),
    )
    point_set.update_points_inspection_status(camera, sun, binary_ray)
    total_weight_inspected = point_set.get_total_weight_inspected(
        inspector_entity=camera
    )
    expected_total_weight_inspected = sum(
        [
            p.weight
            for p in point_set.points.values()
            if p.inspected and p.inspector == camera
        ]
    )
    assert total_weight_inspected == expected_total_weight_inspected


@pytest.mark.parametrize(
    "camera, sun, binary_ray",
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
            False,
        ),
    ],
)
def get_total_weight_inspected(camera, sun, binary_ray):
    point_set = safe_autonomy_simulation.sims.inspection.InspectionPointSet(
        name="set",
        parent=safe_autonomy_simulation.entities.Point(
            "parent", position=np.array([0, 0, 0])
        ),
        num_points=10,
        radius=1,
        priority_vector=np.array([1, 0, 0]),
    )
    point_set.update_points_inspection_status(camera, sun, binary_ray)
    total_weight_inspected = point_set.get_total_weight_inspected()
    expected_total_weight_inspected = sum(
        [p.weight for p in point_set.points.values() if p.inspected]
    )
    assert total_weight_inspected == expected_total_weight_inspected
