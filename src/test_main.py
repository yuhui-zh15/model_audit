import click


@click.command()
@click.option("--name", prompt="Your name", help="The person to greet.")
def hello(name):
    print(f"hello {name}!")
    return 0


def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 5


if __name__ == "__main__":
    hello()
