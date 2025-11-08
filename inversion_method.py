# This program is part of the ORIS Tool.
# Copyright (C) 2011-2025 The ORIS Authors.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np

def sample_from_hyper_exp(p1, p2, lambda1, lambda2):
    """
    Sample from a hyper-exponential distribution with two components.
    
    Args:
        p1 (float): Probability of the first component.
        p2 (float): Probability of the second component.
        lambda1 (float): Rate parameter of the first component.
        lambda2 (float): Rate parameter of the second component.
        
    Returns:
        float: A sample from the hyper-exponential distribution.
    """
    u = np.random.uniform(0, 1)
    if u < p1:
        return np.random.exponential(1/lambda1)
    else:
        return np.random.exponential(1/lambda2)
    
def sample_generalized_erlang(lambdas):
    """
    Sample from a Generalized Erlang distribution.
    
    Parameters:
    lambdas (list or array): A list of rate parameters for the exponential distributions.
    
    Returns:
    float: A sample from the Generalized Erlang distribution.
    """
    samples = [np.random.exponential(1 / lam) for lam in lambdas]
    
    return sum(samples)