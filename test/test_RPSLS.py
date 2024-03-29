'''Unit tests'''

from safe_autonomy_simulation.RPSLS import rock_choice
from safe_autonomy_simulation.RPSLS import paper_choice
from safe_autonomy_simulation.RPSLS import scissors_choice
from safe_autonomy_simulation.RPSLS import lizard_choice
from safe_autonomy_simulation.RPSLS import spock_choice

def test_rockChoice():
    assert rock_choice(1) == 13 and rock_choice(2) == 12 and rock_choice(5) == 12 and rock_choice(3) == 11 and rock_choice(4) ==11

def test_paperChoice():
    assert paper_choice(1) == 11 and paper_choice(5) == 11 and paper_choice(2)==13 and paper_choice(3)==12 and paper_choice(4)==12

def test_scissorsChoice():
    assert scissors_choice(1)==12  and scissors_choice(5)==12 and scissors_choice(2)==11 and scissors_choice(4)==11 and scissors_choice(3)==13

def test_lizardChoice():
    assert lizard_choice(5)==11 and lizard_choice(2)==11 and lizard_choice(1)==12 and lizard_choice(3)==12 and lizard_choice(4)==13

def test_spockChoice():
    assert spock_choice(1)==11 and spock_choice(3)==11 and spock_choice(2)==12 and spock_choice(4)==12 and spock_choice(5)==13
    