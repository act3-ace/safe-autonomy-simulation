import random

# Global named constants for game choices
ROCK = 1
PAPER = 2
SCISSORS = 3
LIZARD = 4
SPOCK = 5

# Global named constants for game outcomes
PLAYER1_WINS = 11
PLAYER2_WINS = 12
DRAW = 13
ERROR = -1


def text_to_number(choice):
    if choice.lower() == "rock":
        return ROCK
    elif choice.lower() == "paper":
        return PAPER
    elif choice.lower() == "scissors":
        return SCISSORS
    elif choice.lower() == "lizard":
        return LIZARD
    elif choice.lower() == "spock":
        return SPOCK
    else:
        return ERROR


def is_valid(choice):
    return choice.lower() in ["rock", "paper", "scissors", "lizard", "spock"]


def rock_choice(player2):
    if player2 == ROCK:
        return DRAW
    elif player2 in [PAPER, SPOCK]:
        return PLAYER2_WINS
    elif player2 in [SCISSORS, LIZARD]:
        return PLAYER1_WINS
    else:
        return ERROR


def paper_choice(player2):
    if player2 in [ROCK, SPOCK]:
        return PLAYER1_WINS
    elif player2 == PAPER:
        return DRAW
    elif player2 in [SCISSORS, LIZARD]:
        return PLAYER2_WINS
    else:
        return ERROR


def scissors_choice(player2):
    if player2 in [ROCK, SPOCK]:
        return PLAYER2_WINS
    elif player2 in [PAPER, LIZARD]:
        return PLAYER1_WINS
    elif player2 == SCISSORS:
        return DRAW
    else:
        return ERROR


def lizard_choice(player2):
    if player2 in [SPOCK, PAPER]:
        return PLAYER1_WINS
    elif player2 in [ROCK, SCISSORS]:
        return PLAYER2_WINS
    elif player2 == LIZARD:
        return DRAW
    else:
        return ERROR


def spock_choice(player2):
    if player2 in [SCISSORS, ROCK]:
        return PLAYER1_WINS
    elif player2 in [LIZARD, PAPER]:
        return PLAYER2_WINS
    elif player2 == SPOCK:
        return DRAW
    else:
        return ERROR


def main():
    player1 = input(
        "Player 1, enter your choice (rock, paper, scissors, lizard, Spock): "
    )

    while not is_valid(player1):
        print("Invalid choice. Please enter a valid choice.")
        player1 = input(
            "Player 1, enter your choice (rock, paper, scissors, lizard, Spock): "
        )

    player1 = text_to_number(player1)

    player2 = random.randint(1, 5)

    if player1 == ROCK:
        result = rock_choice(player2)
    elif player1 == PAPER:
        result = paper_choice(player2)
    elif player1 == SCISSORS:
        result = scissors_choice(player2)
    elif player1 == LIZARD:
        result = lizard_choice(player2)
    elif player1 == SPOCK:
        result = spock_choice(player2)
    else:
        result = ERROR

    if result == PLAYER2_WINS:
        print("Player 2 (computer) wins!")
    elif result == PLAYER1_WINS:
        print("Player 1 wins!")
    elif result == DRAW:
        print("It's a draw!")
    else:
        print("Error occurred.")


if __name__ == "__main__":
    main()
