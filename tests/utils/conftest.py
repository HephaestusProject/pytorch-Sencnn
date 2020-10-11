import os
from pathlib import Path

import pytest


# 각각의 module에서 공통적으로 사용하는 fixture의 경우 conftest.py에 정의하고 사용한다.
# setup_package()를 생각한 기능모사
@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown_package():
    test_dir = Path.cwd() / "tests"
    nsmc_samples_filepath = test_dir / "nsmc_samples.txt"
    trec6_samples_filepath = test_dir / "trec6_samples.txt"

    list_of_nsmc_samples = [
        "document\tlabel\n",
        "재미없을거 같네요 하하하\t0\n정말 좋네요 이 영화 강추!!!!\t1",
    ]
    list_of_trec6_samples = [
        "document\tlabel\n",
        "How big is a quart ?\t5\n",
        "How old is Jeremy Piven ?\t5",
    ]

    with open(nsmc_samples_filepath, mode="w", encoding="utf-8") as io:
        for nsmc_sample in list_of_nsmc_samples:
            io.write(nsmc_sample)

    with open(trec6_samples_filepath, mode="w", encoding="utf-8") as io:
        for trec6_sample in list_of_trec6_samples:
            io.write(trec6_sample)

    yield

    os.remove(nsmc_samples_filepath)
    os.remove(trec6_samples_filepath)


@pytest.fixture(scope="package")
def filepath_of_each_samples():
    test_dir = Path.cwd() / "tests"
    nsmc_samples_filepath = test_dir / "nsmc_samples.txt"
    trec6_samples_filepath = test_dir / "trec6_samples.txt"

    class SamplesPaths:
        pass

    samples_paths = SamplesPaths
    samples_paths.nsmc_samples_filepath = nsmc_samples_filepath
    samples_paths.trec6_samples_filepath = trec6_samples_filepath
    return samples_paths
