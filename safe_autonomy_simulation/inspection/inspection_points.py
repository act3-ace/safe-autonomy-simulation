import copy
import math
import typing

import numpy as np

from pydantic import BaseModel, validator
from safe_autonomy_simulation.base_models import BaseEntity

from safe_autonomy_simulation.spacecraft import CWHSpacecraft, SixDOFSpacecraft
from sklearn.cluster import KMeans

import safe_autonomy_simulation.inspection.illumination as illum


class InspectionPointsValidator(BaseModel):
    """
    Validator for an InspectionPoints object.

    num_points: int
        The number of inspectable points maintained.
    radius: float
        The radius of the sphere on which the points will be generated.
    points_algorithm: str
        The name of the algorithm used to generate initial point positions.
    sensor_fov: float
        The field of view of the inspector's camera sensor in radians.
    initial_sensor_unit_vec: list
        The initial direction the inspector's camera sensor is pointing.
    illumination_params: typing.Union[IlluminationValidator, None]
        The parameters defining lighting of the environment.
    """

    num_points: int
    radius: float
    points_algorithm: str = "cmu"
    sensor_fov: float = np.pi
    initial_sensor_unit_vec: list = [1.0, 0.0, 0.0]
    illumination_params: typing.Union[illum.IlluminationParams, None] = None

    @validator("points_algorithm")
    def valid_algorithm(cls, v):
        """
        Check if provided algorithm is a valid choice.
        """
        valid_algs = ["cmu", "fibonacci"]
        if v not in valid_algs:
            raise ValueError(f"field points_algorithm must be one of {valid_algs}")
        return v


class InspectionPoints:
    """
    A class maintaining the inspection status of an entity.
    """

    def __init__(
        self, parent_entity: CWHSpacecraft, priority_vector: np.ndarray, **kwargs
    ):
        self.config: InspectionPointsValidator = self.get_validator(**kwargs)
        self.sun_angle = 0.0
        self.clock = 0.0
        self.parent_entity = parent_entity
        self.priority_vector = priority_vector
        (
            self._default_points_position_dict,
            self.points_position_dict,
            self.points_inspected_dict,
            self.points_weights_dict,
        ) = self._add_points()
        self.last_points_inspected = 0
        self.last_cluster = None

    @property
    def get_validator(self) -> typing.Type[InspectionPointsValidator]:
        """
        Get the validator used to validate the kwargs passed to BaseAgent.

        Returns
        -------
        BaseAgentParser
            A BaseAgent kwargs parser and validator.
        """
        return InspectionPointsValidator

    def _add_points(self):
        """
        Generate a map of inspection point coordinates to inspected state.

        Returns
        -------
        points_dict
            dict of points_dict[cartesian_point] = initial_inspected_state
        """
        if self.config.points_algorithm == "cmu":
            points_alg = self.points_on_sphere_cmu
        else:
            points_alg = self.points_on_sphere_fibonacci
        points = points_alg(
            self.config.num_points, self.config.radius
        )  # TODO: HANDLE POSITION UNITS*
        points_position_dict = {}
        points_inspected_dict = {}
        points_weights_dict = {}
        for i, point in enumerate(points):
            points_position_dict[i] = point
            points_inspected_dict[i] = False
            points_weights_dict[i] = (
                np.arccos(
                    np.dot(-self.priority_vector, point)
                    / (np.linalg.norm(-self.priority_vector) * np.linalg.norm(point))
                )
                / np.pi
            )

        # Normalize weighting
        total_weight = sum(list(points_weights_dict.values()))
        points_weights_dict = {
            k: w / total_weight for k, w in points_weights_dict.items()
        }

        default_points_position = copy.deepcopy(points_position_dict)

        return (
            default_points_position,
            points_position_dict,
            points_inspected_dict,
            points_weights_dict,
        )

    # inspected or not
    def update_points_inspection_status(self, inspector_entity):
        """
        Update the inspected state of all inspection points given an inspector's position.

        Parameters
        ----------
        position: tuple or array
            inspector's position in cartesian coords

        Returns
        -------
        None
        """
        # calculate h of the spherical cap (inspection zone)
        position = inspector_entity.position
        if isinstance(inspector_entity, SixDOFSpacecraft):
            r_c = inspector_entity.orientation.apply(
                self.config.initial_sensor_unit_vec
            )
        else:
            r_c = -position
        r_c = r_c / np.linalg.norm(r_c)

        r = self.config.radius
        rt = np.linalg.norm(position)
        h = 2 * r * ((rt - r) / (2 * rt))

        p_hat = position / np.linalg.norm(
            position
        )  # position unit vector (inspection zone cone axis)

        for (
            point_id,
            point_position,
        ) in (
            self.points_position_dict.items()
        ):  # pylint: disable=too-many-nested-blocks
            # check that point hasn't already been inspected
            if not self.points_inspected_dict[point_id]:
                p = point_position - position
                p_rc = np.dot(p, r_c) * r_c
                d = np.linalg.norm(p - p_rc)
                c_r = np.linalg.norm(p_rc) * np.tan(self.config.sensor_fov / 2)
                if c_r >= d:
                    # if no illumination params detected
                    if not self.config.illumination_params:
                        # project point onto inspection zone axis and check if in inspection zone
                        if np.dot(point_position, p_hat) >= r - h:
                            self.points_inspected_dict[point_id] = inspector_entity.name
                    else:
                        mag = np.dot(point_position, p_hat)
                        if mag >= r - h:
                            r_avg = self.config.illumination_params.avg_rad_Earth2Sun
                            chief_properties = (
                                self.config.illumination_params.chief_properties
                            )
                            light_properties = (
                                self.config.illumination_params.light_properties
                            )
                            current_theta = self.sun_angle
                            if self.config.illumination_params.bin_ray_flag:
                                if illum.check_illum(
                                    point_position, current_theta, r_avg, r
                                ):
                                    self.points_inspected_dict[
                                        point_id
                                    ] = inspector_entity.name
                            else:
                                RGB = illum.compute_illum_pt(
                                    point_position,
                                    current_theta,
                                    position,
                                    r_avg,
                                    r,
                                    chief_properties,
                                    light_properties,
                                )
                                if illum.evaluate_RGB(RGB):
                                    self.points_inspected_dict[
                                        point_id
                                    ] = inspector_entity.name

    def kmeans_find_nearest_cluster(self, position):
        """Finds nearest cluster of uninspected points using kmeans clustering"""
        uninspected = []
        for point_id, inspected in self.points_inspected_dict.items():
            point_position = self.points_position_dict[point_id]
            if not inspected:
                if self.config.illumination_params:
                    if self.check_if_illuminated(point_position, position):
                        uninspected.append(point_position)
                else:
                    uninspected.append(point_position)
        if len(uninspected) == 0:
            out = np.array([0.0, 0.0, 0.0])
        else:
            n = math.ceil(len(uninspected) / 10)
            data = np.array(uninspected)
            if self.last_cluster is None:
                init = np.zeros((n, 3))
            else:
                if n > self.last_cluster.shape[0]:
                    idxs = np.random.choice(
                        self.last_cluster.shape[0], size=n - self.last_cluster.shape[0]
                    )
                    new = np.array(uninspected)[idxs, :]
                    init = np.vstack((self.last_cluster, new))
                else:
                    init = self.last_cluster[0:n, :]
            kmeans = KMeans(
                init=init,
                n_clusters=n,
                n_init=10,
                max_iter=50,
            )
            kmeans.fit(data)
            self.last_cluster = kmeans.cluster_centers_
            dist = []
            for center in self.last_cluster:
                dist.append(np.linalg.norm(position - center))
            out = kmeans.cluster_centers_[np.argmin(dist)]
            out = out / np.linalg.norm(out)
        return out

    def check_if_illuminated(self, point, position):
        """Check if points is illuminated"""
        r = self.config.radius
        r_avg = self.config.illumination_params.avg_rad_Earth2Sun
        chief_properties = self.config.illumination_params.chief_properties
        light_properties = self.config.illumination_params.light_properties
        current_theta = self.sun_angle
        if self.config.illumination_params.bin_ray_flag:
            illuminated = illum.check_illum(point, current_theta, r_avg, r)
        else:
            RGB = illum.compute_illum_pt(
                point,
                current_theta,
                position,
                r_avg,
                r,
                chief_properties,
                light_properties,
            )
            illuminated = illum.evaluate_RGB(RGB)
        return illuminated

    def points_on_sphere_fibonacci(self, num_points: int, radius: float) -> list:
        """
        Generate a set of equidistant points on sphere using the
        Fibonacci Sphere algorithm: https://arxiv.org/pdf/0912.4540.pdf

        Parameters
        ----------
        num_points: int
            number of points to attempt to place on a sphere
        radius: float
            radius of the sphere

        Returns
        -------
        points: list
            Set of equidistant points on sphere in cartesian coordinates
        """
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians

        for i in range(num_points):
            y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
            r = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * r
            z = math.sin(theta) * r

            points.append(radius * np.array([x, y, z]))

        return points

    def points_on_sphere_cmu(self, num_points: int, radius: float) -> list:
        """
        Generate a set of equidistant points on a sphere using the algorithm
        in https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf. Number
        of points may not be exact.

        Mostly the same as CMU algorithm, most important tweak is that the constant "a" should not depend on r
        (Paper assumed r = 1)

        Parameters
        ----------
        num_points: int
            number of points to attempt to place on a sphere
        radius: float
            radius of the sphere

        Returns
        -------
        points: list
            Set of equidistant points on sphere in cartesian coordinates
        """
        points = []

        a = 4.0 * math.pi * (1 / num_points)
        d = math.sqrt(a)
        m_theta = int(round(math.pi / d))
        d_theta = math.pi / m_theta
        d_phi = a / d_theta

        for m in range(0, m_theta):
            theta = math.pi * (m + 0.5) / m_theta
            m_phi = int(round(2.0 * math.pi * math.sin(theta) / d_phi))
            for n in range(0, m_phi):
                phi = 2.0 * math.pi * n / m_phi

                x = radius * math.sin(theta) * math.cos(phi)
                y = radius * math.sin(theta) * math.sin(phi)
                z = radius * math.cos(theta)

                points.append(np.array([x, y, z]))

        return points

    def update_points_position(self):
        """
        Return the new locations of the points on the chief after rotation

        Parameters
        ----------
        points: list
            points on spherical chief
        current_quat:
            current attitude of chief according to propagation with angular velocities

        Returns
        -------
        newPoints: dict
            rotated points on the chief
        """

        # get parent entity info
        parent_position = self.parent_entity.position
        parent_orientation = self.parent_entity.orientation

        for point_id, default_position in self._default_points_position_dict.items():
            # rotate about origin
            new_position = parent_orientation.apply(default_position)
            # translate from origin
            new_position = new_position + parent_position
            self.points_position_dict[point_id] = new_position

    # getters / setters
    def get_num_points_inspected(self, inspector_entity: BaseEntity = None):
        """Get total number of points inspected"""
        num_points = 0
        if inspector_entity:
            # count number of points inspected by the provided entity
            for _, point_inspector_entity in self.points_inspected_dict.items():
                num_points += (
                    1 if point_inspector_entity == inspector_entity.name else 0
                )
        else:
            # count the total number of points inspected
            for _, point_inspector_entity in self.points_inspected_dict.items():
                num_points += 1 if point_inspector_entity else 0

        return num_points

    def get_percentage_of_points_inspected(self, inspector_entity: BaseEntity = None):
        """Get the percentage of points inspected"""
        total_num_points = len(self.points_inspected_dict.keys())

        if inspector_entity:
            percent = (
                self.get_num_points_inspected(inspector_entity=inspector_entity)
                / total_num_points
            )
        else:
            percent = self.get_num_points_inspected() / total_num_points
        return percent

    def get_cluster_location(self, inspector_position):
        """Get the location of the nearest cluster of uninspected points"""
        return self._kmeans_find_nearest(inspector_position)

    def get_total_weight_inspected(self, inspector_entity: BaseEntity = None):
        """Get total weight of points inspected"""
        weights = 0
        if inspector_entity:
            for point_inspector_entity, weight in zip(
                self.points_inspected_dict.values(), self.points_weights_dict.values()
            ):
                weights += (
                    weight if point_inspector_entity == inspector_entity.name else 0.0
                )
        else:
            for point_inspector_entity, weight in zip(
                self.points_inspected_dict.values(), self.points_weights_dict.values()
            ):
                weights += weight if point_inspector_entity else 0.0
        return weights

    def set_sun_angle(self, sun_angle: np.ndarray):
        """Get the current sun angle"""
        self.sun_angle = float(sun_angle)
