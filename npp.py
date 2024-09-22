# Copyright 2020 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy

## ------- import packages -------
from dwave.system import DWaveSampler, EmbeddingComposite
from random import choice
from functools import reduce, partial

# TODO:  Add code here to define your QUBO dictionary
def set_qubo(nums_and_parition, coords):
    """
    Args:
        nums_and_parition: a map of key-val pairs of a number and the partition the number pertains to
        coords: coordinates for a qubo matrix

    Returns: a qubo containing the coordinates in coords with their respective value under the objective function
    """
    def add_point(num_sum, partial_qubo, point):
        """
        Args:
            num_sum: the sum of all the numbers being partitioned
            partial_qubo: the qubo being filled
            point: the point to add

        Returns: a qubo containing the new point along with its associated value under the objective function

        """
        i, j = point
        cpy = copy.deepcopy(partial_qubo)
        if i < j:
            cpy[(i, j)] = 8 * i * j
        else:
            cpy[(i, i)] = 4 * i * (-num_sum + i)

        return cpy

    add_point_using_sum = partial(add_point, sum(nums_and_parition.keys()))
    return reduce(add_point_using_sum, coords, {})

def get_qubo(nums):
    """Returns a dictionary representing a QUBO.

    Args:
        nums(list of integers): represents the numbers being partitioned
    """
    num_to_partition = {num: choice([0, 1]) for num in nums}
    coords_for_qubo = [(num1, num2) for num1 in nums for num2 in nums if num1 <= num2]

    return set_qubo(num_to_partition, coords_for_qubo)

# TODO:  Choose QPU parameters in the following function
def run_on_qpu(Q, sampler):
    """Runs the QUBO problem Q on the sampler provided.

    Args:
        Q(dict): a representation of a QUBO
        sampler(dimod.Sampler): a sampler that uses the QPU
    """

    chainstrength = 2890 # update
    numruns = 100 # update

    sample_set = sampler.sample_qubo(Q, num_reads=numruns, label='Training - Number Partitioning')

    return sample_set


## ------- Main program -------
if __name__ == "__main__":

    ## ------- Set up our list of numbers -------
    S = [25, 7, 13, 31, 42, 17, 21, 10]

    ## ------- Set up our QUBO dictionary -------

    Q = get_qubo(S)

    ## ------- Run our QUBO on the QPU -------

    sampler = EmbeddingComposite(DWaveSampler())

    sample_set = run_on_qpu(Q, sampler)

    ## ------- Return results to user -------
    for sample in sample_set:
        S1 = [i for i in sample if sample[i] == 1]
        S0 = [i for i in sample if sample[i] == 0]
        print("S0 Sum: ", sum(S0), "\tS1 Sum: ", sum(S1), "\t", S0)

