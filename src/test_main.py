import click


@click.command()
@click.option("--name", prompt="Your name", help="The person to greet.")
def hello(name: str) -> int:
    print(f"hello {name}!")
    return 0


def inc(x: int) -> int:
    return x + 1


def test_answer() -> None:
    assert inc(4) == 5


if __name__ == "__main__":
    hello()
    inc("a")  # type: ignore
