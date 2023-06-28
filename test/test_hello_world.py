'''Unit tests'''

from safe-autonomy-simulation.hello_world import hello_world


def test_answer():
    assert hello_world() == "Hello, world!"
