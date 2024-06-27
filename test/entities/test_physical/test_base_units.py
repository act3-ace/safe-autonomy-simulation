import pint
import pytest
import safe_autonomy_simulation
import safe_autonomy_simulation.entities.physical


STR_UNIT_GROUPS = [
    (d, t, a)
    for d in ["m", "km", "ft", "mi"]
    for t in ["s", "min", "hr"]
    for a in ["rad", "deg", "cycle"]
]
PINT_UNIT_GROUPS = [
    (pint.Unit(d), pint.Unit(t), pint.Unit(a)) for d, t, a in STR_UNIT_GROUPS
]
UNIT_GROUPS = STR_UNIT_GROUPS + PINT_UNIT_GROUPS


@pytest.mark.parametrize(
    "distance, time, angle",
    UNIT_GROUPS,
)
def test_init(distance, time, angle):
    base_units = safe_autonomy_simulation.entities.physical.BaseUnits(
        distance, time, angle
    )
    assert base_units.distance == pint.Unit(distance)
    assert base_units.time == pint.Unit(time)
    assert base_units.angle == pint.Unit(angle)
    assert base_units.velocity == pint.Unit(distance) / pint.Unit(time)
    assert base_units.angular_velocity == pint.Unit(angle) / pint.Unit(time)
    assert base_units.acceleration == pint.Unit(distance) / (pint.Unit(time) ** 2)
    assert base_units.angular_acceleration == pint.Unit(angle) / (pint.Unit(time) ** 2)
