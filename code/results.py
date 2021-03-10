from __future__ import annotations
from collections import namedtuple
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import json



"""Encodes any np.array as a list so json can handle."""
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



"""
A Result stores a detailed description of an optimization result.

Will cast as dict for dumping, but in computation I use tuples (immutable) for safety.
"""
Result = namedtuple('Result',
                    ['state', 'detpoints', 'weights', 'optval'])


class Results():
    """
    A 'Results' instance has a list of 'Result' tuples, and has auxiliary methods
    for appending more results, plotting data, exporting to files etc.
    """

    def __init__(self, measurements: Optional[np.ndarray] = None,
                 parameters: Optional[Union[List, np.ndarray]] = None,
                 description: Optional[str] = None,
                 results: Optional[List] = [],
                 fname: Optional[str] = None):
        """
        measurements: an array to register which measurements were used.
        parameters: parameters related to each Result, for instance a meshgrid (helps plotting).
        description: a description of the results for future reference.
        results: list of all the 'Result' instances.
        fname: if you want to load previous results from a file.
        """

        if fname:
            measurements, parameters, description, results = self.load(fname)

        self.measurements = measurements
        self.results = results
        self.parameters = parameters
        self.description = description


    def __repr__(self):
        return (self.description + "\nStored results: " + str(len(self.results)))

    def __str__(self):
        self.__repr__()

    def __eq__(self, other):
        # TODO
        pass

    # TODO: Deixar indexar direto na instancia inves de em self.results.


    @property
    def optvals(self) -> List:
        """List with the optvals of all instances."""
        return [self.results[i].optval for i in range(len(self.results))]


    def append(self, result: Result) -> None:
        """Append a new Result."""

        self.results.append(result)


    def save(self, fname: str = str(datetime.now())) -> None:
        """Export self to fname as a json file, overwriting fname if it exists.

        NumpyEncoder will convert any np.ndarray data to (possibly nested) lists.

        A more straightforward alternative is to just pickle to object, but it could
        lead to compatibility issues and wouldn't be loadable elsewhere."""

        serialized = {'measurements': self.measurements,
                      'description': self.description,
                      'parameters': self.parameters,
                      'results': [{'state': result.state,
                                   'detpoints': result.detpoints,
                                   'weights': result.weights,
                                   'optval': result.optval}
                                  for result in self.results]}

        with open(fname + '.json', 'w') as file:
            json.dump(serialized, file, cls=NumpyEncoder)


    def load(self, fname: str) -> Results:
        """Reload a Results object from fname.
        (Could also extend a json.JSONDecoder with an object_hook to do this...)

        I'm not recasting each Result's fields to np.arrays as np functions
        will naturally do whenever called.

        # TODO: Num vai não. Deu problema no reshape da contourplot. Já converter
        """

        with open(fname, 'r') as file:
            ld = json.load(file)

        for i in range(len(ld['results'])):
            ld['results'][i] = Result(**ld['results'][i])
        return ld['measurements'], ld['parameters'], ld['description'], ld['results']


    def plot(self, **kwargs):
        """Return a 2D or a countour plot, depending on which parametrization was done."""

        def plot2d(self, X, Y, grid=False, marker=None, legend=None, ax=None):
            """Simple y(x) plot."""

            if not ax: ax = plt
            line = ax.plot(X, Y, linewidth=2, markersize=5, marker=marker)

            if grid: ax.grid(linestyle='dashed', color='gray', alpha=.5)
            if legend: line.set_label(self.measurements.meastag)


        def contour_plot(self, X, Y, Z, levels=10, cmap=cm.Blues_r,
                         edges=False, grid_points=False, ax=None):
            """Contour map with interpolated z(x, y) function."""

            if not ax: ax = plt
            contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
            ax.colorbar(contour)

            if edges: ax.contour(X, Y, Z, levels=levels, linewidths=.5,
                                 alpha=.5, colors="black")
            if grid_points: ax.scatter(X, Y, c=Z, s=8, cmap=cmap)


        # --------------------------

        try:
            X, Y = np.asarray(self.parameters)
            contour_plot(self, X, Y, np.reshape(self.optvals, X.shape), **kwargs)
        except (TypeError, ValueError, AttributeError):
            plot2d(self, self.parameters, self.optvals, **kwargs)

