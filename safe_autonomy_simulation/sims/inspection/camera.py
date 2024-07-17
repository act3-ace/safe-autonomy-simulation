"""Camera entity for capturing images of objects in the environment"""

import numpy as np
import scipy.spatial.transform as transform
import safe_autonomy_simulation.entities as e
import safe_autonomy_simulation.dynamics as d
import safe_autonomy_simulation.controls as c
import safe_autonomy_simulation.materials as m
import safe_autonomy_simulation.sims.inspection.utils.illumination as illum
import safe_autonomy_simulation.sims.inspection.utils.vector as vector
import safe_autonomy_simulation.sims.inspection.utils.sphere as sphere


class Camera(e.PhysicalEntity):
    """Camera entity that can capture images of objects in the environment

    Parameters
    ----------
    name: str
        name of the entity
    fov: float
        field of view of the camera in radians
    resolution: list[int]
        resolution of the camera in pixels
    focal_length: float
        focal length of the camera in meters
    pixel_pitch: float
        pixel pitch of the camera in meters
    position: np.ndarray, optional
        initial absolute position of the entity, by default [0, 0, 0]
    velocity: np.ndarray, optional
        initial absolute velocity of the entity, by default [0, 0, 0]
    orientation: Rotation, optional
        initial absolute orientation quaternion of the entity, by default [0, 0, 0, 1]
    angular_velocity: np.ndarray, optional
        initial absolute angular velocity of the entity, by default [0, 0, 0]
    dynamics: Dynamics, optional
        dynamics of the entity, by default PassThroughDynamics()
    parent: PhysicalEntity, optional
        parent entity, by default None
    children: list[Entity], optional
        list of children entities, by default []
    """

    def __init__(
        self,
        name: str,
        fov: float,
        resolution: list[int],
        focal_length: float,
        pixel_pitch: float,
        position: np.ndarray = np.zeros(3),
        velocity: np.ndarray = np.zeros(3),
        orientation: np.ndarray = transform.Rotation.from_euler(
            "ZYX", [0, 0, 0]
        ).as_quat(),
        angular_velocity: np.ndarray = np.zeros(3),
        dynamics: d.Dynamics = d.PassThroughDynamics(),
        control_queue: c.ControlQueue = c.NoControl(),
        material: m.Material = m.BLACK,
        parent: e.PhysicalEntity = None,
        children: list[e.Entity] = [],
    ):
        super().__init__(
            name=name,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            dynamics=dynamics,
            control_queue=control_queue,
            material=material,
            parent=parent,
            children=children,
        )
        self._fov = fov
        self._resolution = resolution
        self._focal_length = focal_length
        self._pixel_pitch = pixel_pitch

    # def point_at(self, target: e.PhysicalEntity):
    #     """Point the camera at a target

    #     Sets the camera orientation to point at the target
    #     entity's position.

    #     Parameters
    #     ----------
    #     target: Entity
    #         target to point the camera at
    #     """
    #     target_position = target.position
    #     target_direction = vector.normalize(target_position - self.position)
    #     new_orientation = transform.Rotation.from_euler(
    #         "xyz", target_direction
    #     ).as_quat()
    #     self.state = np.concatenate(
    #         (self.position, self.velocity, new_orientation, self.angular_velocity)
    #     )

    def check_point_illumination(
        self,
        point: e.Point,
        light: e.PhysicalEntity,
        viewed_object: e.Entity,
        radius: float,
        binary_ray: bool = False,
    ) -> bool:
        """Check if point is illuminated

        Parameters
        ----------
        point: Point
            point to check for illumination
        light: PhysicalEntity
            light entity
        viewed_object: Entity
            object on which the point is located
        radius: float
            radius of the viewed object in meters
        binary_ray: bool, optional
            whether to use binary ray tracing for illumination, by default False

        Returns
        -------
        bool
            point illumination status, True if illuminated, False if not
        """
        if binary_ray:
            illuminated = illum.is_illuminated(point=point, light=light, radius=radius)
        else:
            rgb = self.capture_point(
                point=point,
                light=light,
                viewed_object=viewed_object,
                radius=radius,
            )
            illuminated = illum.is_illuminated_rgb(rgb)
        return illuminated

    def capture_point(
        self,
        point: e.Point,
        light: e.PhysicalEntity,
        viewed_object: e.Entity,
        radius: float,
    ) -> np.ndarray:
        """Capture a point on an object in the environment

        Parameters
        ----------
        point: Point
            point on the object to capture
        light: PhysicalEntity
            light source entity
        viewed_object: Entity
            object to capture the point from
        radius: float
            radius of the object in meters

        Returns
        -------
        np.ndarray
            rgb pixel value of the point
        """
        # Chief position is origin [cwh dynamics]
        # TODO: don't assume origin
        center = [0, 0, 0]
        normal_to_surface = vector.normalize(point.position)
        # Get a point slightly off the surface of the sphere so don't detect surface as an intersection
        shifted_point = point.position + 1e-5 * normal_to_surface
        intersection_to_light = vector.normalize(light.position - shifted_point)
        intersect_var = sphere.sphere_intersect(
            center, radius, shifted_point, intersection_to_light
        )

        illumination = np.zeros((3))
        # No intersection means that the point in question is illuminated in some capacity
        # (i.e. the point on the chief is not blocked by the chief itself)
        if intersect_var is None:
            # Blinn-Phong Illumination Model
            # https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model
            illumination += np.array(viewed_object.material.ambient) * np.array(
                light.material.ambient
            )
            illumination += (
                np.array(viewed_object.material.diffuse)
                * np.array(light.material.diffuse)
                * np.dot(intersection_to_light, normal_to_surface)
            )
            intersection_to_camera = vector.normalize(self.position - point)
            H = vector.normalize(intersection_to_light + intersection_to_camera)
            illumination += (
                np.array(viewed_object.material.specular)
                * np.array(light.material.specular)
                * np.dot(normal_to_surface, H) ** (viewed_object.material.shininess / 4)
            )
            illumination = np.clip(illumination, 0, 1)
        return illumination

    def capture(
        self, light: e.PhysicalEntity, viewed_object: e.PhysicalEntity, radius: float
    ) -> np.ndarray:
        """Capture an image of an object in the environment

        Parameters
        ----------
        light: PhysicalEntity
            light source entity
        viewed_object: PhysicalEntity
            object to capture the image from
        radius: float
            radius of the object in meters

        Returns
        -------
        np.ndarray
            rgb image of the object
        """
        ratio = float(self.resolution[0]) / self.resolution[1]
        # For now, assuming deputy sensor always pointed at chief (which is origin)
        # TODO: don't assume origin
        chief_position = [0, 0, 0]
        sensor_dir = vector.normalize(chief_position - self.position)
        image_plane_position = self.position + sensor_dir * self.focal_length
        # There are an infinite number of vectors normal to sensor_dir -- choose one
        x = -1
        y = 1
        z = (
            -(image_plane_position[0] * x + image_plane_position[1] * y)
            / image_plane_position[2]
        )
        norm1 = vector.normalize([x, y, z])

        # np.cross bug work-around https://github.com/microsoft/pylance-release/issues/3277
        def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.cross(a, b)

        norm2 = cross2(sensor_dir, norm1)
        # Used for x,y,z pixel locations - there will be resolution[0] * resolution[1] pixels
        x_width = (
            np.tan((self.pixel_pitch / self.focal_length) / 2) * 2 * self.focal_length
        )
        norm1_range = x_width
        norm2_range = x_width / ratio
        step_norm1 = norm1_range / (self.resolution[0])
        step_norm2 = norm2_range / (self.resolution[1])
        # 3D matrix (ie. height-by-width matrix with each entry being an array of size 3) which creates an image
        image = np.zeros((self.resolution[1], self.resolution[0], 3))
        for i in range(int(self.resolution[1])):  # y coords
            for j in range(int(self.resolution[0])):  # x coords
                # Initialize pixel
                illumination = np.zeros((3))
                # Convert to CWH coordinates
                pixel_location = (
                    image_plane_position
                    + ((norm2_range / 2) - (i * step_norm2)) * (norm2)
                    + (-(norm1_range / 2) + (j * step_norm1)) * (norm1)
                )
                ray_direction = vector.normalize(pixel_location - self.position)
                dist_2_intersect = sphere.sphere_intersect(
                    chief_position, radius, self.position, ray_direction
                )
                # Light ray hits sphere, so we continue - else get next pixel
                if dist_2_intersect is not None:
                    intersection_point = (
                        self.position + dist_2_intersect * ray_direction
                    )
                    normal_to_surface = vector.normalize(
                        intersection_point - chief_position
                    )
                    shifted_point = intersection_point + 1e-5 * normal_to_surface
                    intersection_to_light = vector.normalize(
                        light.position - shifted_point
                    )
                    intersect_var = sphere.sphere_intersect(
                        chief_position, radius, shifted_point, intersection_to_light
                    )
                    # If the shifted point doesn't intersect with the chief on the way to the light, it is unobstructed
                    if intersect_var is None:
                        # Blinn-Phong Illumination Model
                        # https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model

                        illumination += np.array(
                            viewed_object.material.ambient
                        ) * np.array(light.material.ambient)
                        illumination += (
                            np.array(viewed_object.material.diffuse)
                            * np.array(light.material.diffuse)
                            * np.dot(intersection_to_light, normal_to_surface)
                        )
                        intersection_to_camera = vector.normalize(
                            self.position - intersection_point
                        )
                        H = vector.normalize(
                            intersection_to_light + intersection_to_camera
                        )
                        illumination += (
                            np.array(viewed_object.material.specular)
                            * np.array(light.material.specular)
                            * np.dot(normal_to_surface, H)
                            ** (viewed_object.material.shininess / 4)
                        )
                    # Shadowed
                    else:
                        continue
                image[i, j] = np.clip(illumination, 0, 1)
        return image

    @property
    def fov(self) -> float:
        """Field of view of the camera in degrees

        Returns
        -------
        float
            field of view of the camera in degrees
        """
        return self._fov

    @property
    def resolution(self) -> list[int]:
        """Resolution of the camera in pixels

        Returns
        -------
        list[int]
            resolution of the camera in pixels
        """
        return self._resolution

    @property
    def focal_length(self) -> float:
        """Focal length of the camera in meters

        Returns
        -------
        float
            focal length of the camera in meters
        """
        return self._focal_length

    @property
    def pixel_pitch(self) -> float:
        """Pixel pitch of the camera in meters

        Returns
        -------
        float
            pixel pitch of the camera in meters
        """
        return self._pixel_pitch

    @property
    def state(self) -> np.ndarray:
        """Camera state vector

        Returns
        -------
        np.ndarray
            state vector of the camera
        """
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        assert (
            state.shape == self.state.shape
        ), f"State shape must be {self.state.shape}, got {state.shape}"
        self._state = state
