def pytest_addoption(parser):
    parser.addoption(
        "--num_articles", type=int, default=5, help="number of articles to test"
    )
    parser.addoption("--batch_size", type=int, default=5, help="batch size")


def pytest_generate_tests(metafunc):
    if "num_articles" in metafunc.fixturenames:
        metafunc.parametrize(
            "num_articles", [metafunc.config.getoption("--num_articles")]
        )
    if "batch_size" in metafunc.fixturenames:
        metafunc.parametrize("batch_size", [metafunc.config.getoption("--batch_size")])
