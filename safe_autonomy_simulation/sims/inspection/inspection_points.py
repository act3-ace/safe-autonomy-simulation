import copy
import math
import typing

import numpy as np
import sklearn.cluster as cluster
import scipy.spatial.transform as transform

import safe_autonomy_simulation.entities as e
import safe_autonomy_simulation.dynamics as d
import safe_autonomy_simulation.controls as c
import safe_autonomy_simulation.materials as m
import safe_autonomy_simulation.sims.inspection.camera as cam
import safe_autonomy_simulation.sims.inspection.sun as sun
import safe_autonomy_simulation.sims.inspection.utils.sphere as sphere
import safe_autonomy_simulation.sims.spacecraft as spacecraft


class InspectionPointDynamics(d.Dynamics):
    """Dynamics for an inspection point entity

    Update the position of the point from its default position based on the parent entity's position and orientation.

    Parameters
    ----------
    default_position: np.ndarray
        default absolute position of the point
    parent: PhysicalEntity
        parent entity of the point which the point is anchored to
    """

    def __init__(self, default_position: np.ndarray, parent: e.PhysicalEntity):
        super().__init__()
        self._parent = parent
        self._default_position = default_position

    def _step(
        self, step_size: float, state: np.ndarray, control: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        new_position = transform.Rotation.from_quat(self._parent.orientation).apply(
            self._default_position
        )
        # translate from parent position
        new_position = new_position + self._parent.position
        next_state = np.concatenate((new_position, state[3:]))
        return next_state, control

    @property
    def parent(self) -> e.PhysicalEntity:
        """Parent entity of the point

        Returns
        -------
        PhysicalEntity
            parent entity of the point
        """
        return self._parent

    @property
    def default_position(self) -> np.ndarray:
        """Default position of the point

        Returns
        -------
        np.ndarray
            default position of the point
        """
        return self._default_position


class InspectionPoint(e.Point):
    """A weighted inspection point entity

    Parameters
    ----------
    position: np.ndarray
        absolute position of the point
    inspected: bool
        whether the point has been inspected
    inspector: str
        name of the entity that inspected the point
    weight: float
        weight of the point
    parent: Entity
        parent entity of the point which the point is anchored to
    name: str, optional
        name of the point, by default "point"
    """

    def __init__(
        self,
        position: np.ndarray,
        inspected: bool,
        inspector: str,
        weight: float,
        parent: e.Entity,
        name: str = "point",
    ):
        self._default_position = position
        self._initial_inspected = inspected
        self._inspector = inspector
        self._initial_weight = weight
        super().__init__(
            name=name,
            position=position,
            parent=parent,
            dynamics=InspectionPointDynamics(position, parent),
        )

    def build_initial_state(self) -> np.ndarray:
        # Append weight and inspection status to internal state
        # We keep the internal state vector inherited from PhysicalEntity
        # unchanged so that the PhysicalEntity properties are correct.
        # Our new states are appended to the end of the inherited internal
        # state vector.
        state = super().build_initial_state()
        state = np.concatenate(
            (state, [self._initial_weight], [self._initial_inspected])
        )
        return state

    @property
    def state(self) -> np.ndarray:
        """Inspection point state vector

        Inspection point state vector is [x, y, z, x_dot, y_dot, z_dot, weight, inspected]

        Returns
        -------
        np.ndarray
            inspection point state vector
        """
        return np.concatenate((self._state[:6], [self.weight, self.inspected]))

    @state.setter
    def state(self, state: np.ndarray):
        """Set inspection point state vector

        Parameters
        ----------
        state: np.ndarray
            inspection point state vector
        """
        assert (
            state.shape == self.state.shape
        ), f"State vector must be of shape {self.state.shape}, got {state.shape}"
        self._state[0:6] = state[0:6]
        self.weight = state[6]
        self.inspected = state[7]

    @property
    def default_position(self) -> np.ndarray:
        """Default position of the point

        Returns
        -------
        np.ndarray
            default position of the point
        """
        return self._default_position

    @property
    def inspected(self) -> bool:
        """Inspection status of the point

        Returns
        -------
        bool
            inspection status of the point
        """
        return self._state[-1]

    @inspected.setter
    def inspected(self, inspected: bool):
        self._state[-1] = inspected

    @property
    def inspector(self) -> str:
        """Entity that inspected the point

        Returns
        -------
        str
            entity that inspected the point
        """
        return self._inspector

    @inspector.setter
    def inspector(self, inspector: str):
        self._inspector = inspector

    @property
    def weight(self) -> float:
        """Weight of the point

        Returns
        -------
        float
            weight of the point
        """
        return self._state[-2]

    @weight.setter
    def weight(self, weight: float):
        self._state[-2] = weight


class InspectionPointSet(e.Entity):
    """
    Inspection points entity containing a sphere of inspection points.

    Parameters
    ----------
    name: str
        name of the inspection points entity
    parent: Entity
        parent entity of the inspection points which the points are anchored to
    num_points: int
        number of inspection points
    radius: float
        radius of the sphere of inspection points
    priority_vector: np.ndarray
        priority vector for inspection points weighting
    points_algorithm: str, optional
        algorithm to generate points on sphere, either "cmu" or "fibonacci", by default "cmu"
    half_weighted: bool, optional
        weight only half of the sphere (other half has zero weight). Default false (weight all points)
    """

    def __init__(
        self,
        name: str,
        parent: e.Entity,
        num_points: int,
        radius: float,
        priority_vector: np.ndarray,
        points_algorithm: str = "cmu",
        dynamics: d.Dynamics = d.PassThroughDynamics(),
        control_queue: c.ControlQueue = c.NoControl(),
        material: m.Material = m.BLACK,
        half_weighted: bool = False
    ):
        self.half_weighted = half_weighted
        self._points: typing.Dict[
            int, InspectionPoint
        ] = {}  # initialize points as empty for correct parent assignment
        super().__init__(
            name=name,
            parent=parent,
            dynamics=dynamics,
            control_queue=control_queue,
            material=material,
        )
        self._radius = radius
        self._priority_vector = priority_vector
        self.init_priority_vector = copy.deepcopy(self.priority_vector)
        self._clusters = None
        self._points = self._generate_points(
            num_points=num_points, points_algorithm=points_algorithm
        )

    def build_initial_state(self) -> np.ndarray:
        state = np.array([p.state for p in self.points.values()])
        return state

    def _pre_step(self, step_size: float):
        super()._pre_step(step_size)
        # Update state in pre step to ensure that all points have been updated
        self._state = np.array([p.state for p in self.points.values()])

    def _post_step(self, step_size: float):
        super()._post_step(step_size)
        # Updating state in post step to ensure that all points have been updated
        self._state = np.array([p.state for p in self.points.values()])
        # Update priority vector
        self._priority_vector = transform.Rotation.from_quat(self.parent.orientation).apply(self.init_priority_vector)

    def _generate_points(
        self, num_points, points_algorithm: str = "cmu"
    ) -> typing.Dict[int, InspectionPoint]:
        """Generate a sphere of inspection points

        Depending on which algorithm is chosen, generated points
        on sphere may not equal `num_points`.

        Parameters
        ----------
        num_points: int
            number of inspection points to generate
        points_algorithm: str, optional
            algorithm to generate points on sphere, either "cmu" or "fibonacci", by default "cmu"

        Returns
        -------
        typing.Dict[int, Point]
            dictionary of {point_id: Point}
        """
        assert (
            points_algorithm
            in [
                "cmu",
                "fibonacci",
            ]
        ), f"Invalid points algorithm {points_algorithm}. Must be one of 'cmu' or 'fibonacci'"

        if points_algorithm == "cmu":
            points_alg = sphere.points_on_sphere_cmu
        else:
            points_alg = sphere.points_on_sphere_fibonacci

        # generate point positions
        point_positions = points_alg(
            num_points, self.radius
        )  # TODO: HANDLE POSITION UNITS*
        if not self.half_weighted:
            point_weights = [
                np.arccos(
                    np.dot(-self.priority_vector, pos)
                    / (np.linalg.norm(-self.priority_vector) * np.linalg.norm(pos))
                )
                / np.pi
                for pos in point_positions
            ]
        else:
            point_weights = []
            for pos in point_positions:
                dot_prod = np.dot(-self.priority_vector, pos) / (np.linalg.norm(-self.priority_vector) * np.linalg.norm(pos))
                if dot_prod <= 0:
                    point_weights.append(np.arccos(dot_prod) / np.pi)
                else:
                    point_weights.append(0.0)
        total_weight = sum(point_weights)
        point_weights = [w / total_weight for w in point_weights]  # normalize weights

        points = {}
        for i, (pos, weight) in enumerate(zip(point_positions, point_weights)):
            point = InspectionPoint(
                position=pos,
                inspected=False,
                inspector=None,
                weight=weight,
                parent=self,
            )
            points[i] = point

        return points

    def update_points_inspection_status(
        self,
        camera: cam.Camera,
        sun: typing.Union[sun.Sun, None] = None,
        binary_ray: bool = True,
    ):
        """
        Update the inspected state of all inspection points given an inspector's position.

        If sun entity is given, check if point is illuminated. Otherwise, assume no illumination.

        Parameters
        ----------
        camera: Camera
            camera entity inspecting the points
        sun: Union[Sun, None], optional
            sun entity, by default None
        binary_ray: bool, optional
            whether to use binary ray tracing for illumination, by default True
        """
        # calculate h of the spherical cap (inspection zone)
        cam_position = camera.position
        if isinstance(camera.parent, spacecraft.SixDOFSpacecraft):
            # TODO: orientation axis vector should be defined externally
            r_c = transform.Rotation.from_quat(camera.orientation).apply(
                np.array([1, 0, 0])
            )
        else:
            # For translational motion only, camera always points towards chief (origin)
            # TODO: don't assume chief is at origin
            r_c = -cam_position
        r_c = r_c / np.linalg.norm(r_c)  # inspector sensor unit vector

        r = self.radius
        rt = np.linalg.norm(cam_position)
        h = 2 * r * ((rt - r) / (2 * rt))

        p_hat = cam_position / np.linalg.norm(
            cam_position
        )  # position unit vector (inspection zone cone axis)

        for (
            _,
            point,
        ) in self.points.items():  # pylint: disable=too-many-nested-blocks
            # check that point hasn't already been inspected
            if not point.inspected:
                p = point.position - cam_position
                cos_theta = np.dot(p / np.linalg.norm(p), r_c)
                angle_to_point = np.arccos(cos_theta)
                # If the point can be inspected (within FOV)
                if angle_to_point <= (camera.fov / 2):
                    # if no point light (sun), assume no illumination
                    if not sun:
                        # project point onto inspection zone axis and check if in inspection zone
                        if np.dot(point.position, p_hat) >= r - h:
                            point.inspected = True
                            point.inspector = camera.name
                    else:
                        mag = np.dot(point.position, p_hat)
                        if mag >= r - h and camera.check_point_illumination(
                            point=point,
                            light=sun,
                            viewed_object=self.parent,
                            radius=self.radius,
                            binary_ray=binary_ray,
                        ):
                            point.inspected = True
                            point.inspector = camera.name

    def kmeans_find_nearest_cluster(
        self,
        camera: cam.Camera,
        sun: typing.Union[sun.Sun, None] = None,
        binary_ray: bool = True,
    ) -> np.ndarray:
        """Finds nearest cluster of uninspected points using kmeans clustering

        If sun entity is given, check if point is illuminated. Otherwise, assume no illumination.

        Parameters
        ----------
        camera: Camera
            camera entity inspecting the points
        sun: Union[Sun, None], optional
            sun entity, by default None
        binary_ray: bool, optional
            whether to use binary ray tracing for illumination, by default True

        Returns
        -------
        np.ndarray
            unit vector pointing to nearest cluster
        """
        uninspected = []
        for _, point in self.points.items():
            if not point.inspected and point.weight != 0:
                if sun:
                    if camera.check_point_illumination(
                        point=point,
                        light=sun,
                        viewed_object=self.parent,
                        radius=self.radius,
                        binary_ray=binary_ray,
                    ):
                        uninspected.append(point.position)
                else:
                    uninspected.append(point.position)
        if len(uninspected) == 0:
            out = np.array([0.0, 0.0, 0.0])
        else:
            n = math.ceil(len(uninspected) / 10)
            data = np.array(uninspected)
            if self.clusters is None:
                init = np.zeros((n, 3))
            else:
                if n > self.clusters.shape[0]:
                    idxs = np.random.choice(
                        self.clusters.shape[0], size=n - self.clusters.shape[0]
                    )
                    new = np.array(uninspected)[idxs, :]
                    init = np.vstack((self.clusters, new))
                else:
                    init = self.clusters[0:n, :]
            kmeans = cluster.KMeans(
                init=init,
                n_clusters=n,
                n_init=1,
                max_iter=50,
            )
            kmeans.fit(data)
            self.clusters = kmeans.cluster_centers_
            dist = []
            for center in self.clusters:
                dist.append(np.linalg.norm(camera.position - center))
            out = kmeans.cluster_centers_[np.argmin(dist)]
            out = out / np.linalg.norm(out)
        return out

    def get_num_points_inspected(self, inspector_entity: e.Entity = None) -> int:
        """Get total number of points inspected by an entity.

        If no entity is provided, return total number of points inspected.

        Parameters
        ----------
        inspector_entity: Entity, optional
            entity inspecting the points, by default None

        Returns
        -------
        int
            number of points inspected
        """
        num_points = 0
        for _, point in self.points.items():
            if point.inspected:
                num_points += 1
        return num_points

    def get_percentage_of_points_inspected(
        self, inspector_entity: e.Entity = None
    ) -> float:
        """Get the percentage of points inspected by an entity.

        If no entity is provided, return total percentage of points inspected.

        Parameters
        ----------
        inspector_entity: Entity, optional
            entity inspecting the points, by default None

        Returns
        -------
        float
            percentage of points inspected
        """
        total_num_points = len(self.points)
        percent = self.get_num_points_inspected(inspector_entity) / total_num_points
        return percent

    def get_total_weight_inspected(
        self, inspector_entity: e.Entity | None = None
    ) -> float:
        """Get total weight of points inspected by an entity.

        If no entity is provided, return total weight of points inspected.

        Parameters
        ----------
        inspector_entity: Entity, optional
            entity inspecting the points, by default None

        Returns
        -------
        float
            total weight of points inspected
        """
        weight = 0
        for _, point in self.points.items():
            if inspector_entity:
                weight += (
                    point.weight if point.inspector == inspector_entity.name else 0.0
                )
            else:
                weight += point.weight if point.inspected else 0.0
        return weight

    @property
    def num_points(self) -> int:
        """Number of inspection points

        Returns
        -------
        int
            number of inspection points
        """
        return len(self.points)

    @property
    def radius(self) -> float:
        """Radius of inspection points sphere

        Returns
        -------
        float
            radius of inspection points sphere
        """
        return self._radius

    @property
    def priority_vector(self) -> np.ndarray:
        """Priority vector for inspection points weighting

        Returns
        -------
        np.ndarray
            priority vector for inspection points weighting
        """
        return self._priority_vector

    @property
    def points(self) -> typing.Dict[int, InspectionPoint]:
        """Inspection points dictionary

        Returns
        -------
        typing.Dict[int, InspectionPoint]
            dictionary of {point_id: InspectionPoint}
        """
        return self._points

    @property
    def clusters(self) -> typing.Union[np.ndarray, None]:
        """Clusters of uninspected points

        Returns
        -------
        typing.Union[np.ndarray, None]
            clusters of uninspected points
        """
        return self._clusters

    @clusters.setter
    def clusters(self, clusters: typing.Union[np.ndarray, None]):
        self._clusters = clusters

    @property
    def state(self) -> np.ndarray:
        """Inspection points state vector

        Inspection points state vector is [point_1_state, point_2_state, ..., point_n_state]

        Returns
        -------
        np.ndarray
            inspection points state vector
        """
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        """Set inspection points state vector

        Parameters
        ----------
        state: np.ndarray
            inspection points state vector
        """
        assert (
            state.shape == self.state.shape
        ), f"State vector must be of shape {self.state.shape}, got {state.shape}"
        for i, point in self.points.items():
            point.state = state[i]
        self._state = state

    @property
    def position(self) -> np.ndarray:
        """Position of inspection point set entity

        Returns
        -------
        np.ndarray
            position of inspection point set entity
        """
        return self.parent.position

    @property
    def orientation(self) -> np.ndarray:
        """Orientation of inspection point set entity

        Returns
        -------
        np.ndarray
            orientation of inspection point set entity
        """
        return self.parent.orientation
