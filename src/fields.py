from __future__ import annotations

from copy import deepcopy
from typing import Optional, Union
import warnings

import numpy as np
import matplotlib.pyplot as plt


class ScalarField(np.ndarray):
    """
    Scalar field. The map must be of the kind
        f : ℝ² → ℝ ; (x, y) → F
    """

    EXPECTED_INPUT_DIMENSIONS = [2]

    def __new__(cls, field: Union[ScalarField, np.ndarray]) -> ScalarField:
        """
        Create a new scalar field.

        Parameters
        ----------
        field : Union[ScalarField, np.ndarray]
            A scalar field or a numpy array. The shape of the field must follow this pattern:

                               ┏━ The width of the field (x-axis)
                               ┃  ┏━ The height of the field (y-axis)
                               ┃  ┃
                field.shape = (A, B)

        Returns
        -------
        field : ScalarField
            The new vector field as a numpy array.
        """
        input_dimension = len(field.shape)

        if input_dimension not in cls.EXPECTED_INPUT_DIMENSIONS:
            raise ValueError(f"The input dimension of the given scalar field is not correct. Current dimension is "
                             f"{input_dimension} while accepted dimensions are {cls.EXPECTED_INPUT_DIMENSIONS}.")

        return np.asarray(deepcopy(field)).view(cls)

    @property
    def input_dimension(self) -> int:
        """
        The map's input dimension.

                 ┏━ The dimension of this set.
                 ┃
            f : ℝ² → ℝ.
        """
        return self.ndim

    @property
    def output_dimension(self) -> int:
        """
        The map's output dimension.

                     ┏━ The dimension of this set.
                     ┃
            f : ℝⁿ → ℝ.
        """
        return 1

    def gradient(self) -> VectorField:
        """
        Gradient of the scalar field.

        Returns
        -------
        gradient : VectorField
            The vector field representing the gradient of the current scalar field.
        """
        return VectorField(np.stack(np.gradient(self), axis=2))

    def show(self, **kwargs):
        """
        Show the scalar field in the xy plane.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments to create a custom matplotlib figure. Keyword arguments currently defined are :
                {
                x_label : str
                    Label text for the x-axis.
                y_label : str
                    Label text for the y-axis.
                title : str
                    Text to use for the figure's title.
            }
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        image = ax.imshow(self.T, origin="lower")

        ax.set_xlabel(xlabel=kwargs.get("x_label", "x [m]"))
        ax.set_ylabel(ylabel=kwargs.get("y_label", "y [m]"))
        ax.set_title(label=kwargs.get("title", ""))
        fig.colorbar(image, orientation='vertical')
        plt.show()


class VectorField(np.ndarray):
    """
    Vector field. The two accepted maps forms are
        f : ℝ² → ℝ² ; (x, y) → (F_x, F_y)
    and
        f : ℝ² → ℝ³ ; (x, y) → (F_x, F_y, F_z).
    """

    EXPECTED_INPUT_DIMENSIONS = [2]
    EXPECTED_OUTPUT_DIMENSIONS = [2, 3]

    def __new__(cls, field: Union[VectorField, np.ndarray]) -> VectorField:
        """
        Create a new vector field.

        Parameters
        ----------
        field : Union[VectorField, np.ndarray]
            A vector field or a numpy array. The shape of the field must follow this pattern:

                               ┏━ The width of the field (x-axis)
                               ┃  ┏━ The height of the field (y-axis)
                               ┃  ┃  ┏━ The dimension of the output vector, i.e 2 if (F_x, F_y) and 3 if (F_x, F_y, F_z)
                               ┃  ┃  ┃
                field.shape = (A, B, C)

        Returns
        -------
        field : VectorField
            The new vector field as a numpy array.
        """
        input_dimension = len(field.shape[:-1])
        output_dimension = field.shape[-1]

        if input_dimension not in cls.EXPECTED_INPUT_DIMENSIONS:
            raise ValueError(f"The input dimension of the given vector field is not correct. Current dimension is "
                             f"{input_dimension} while accepted dimensions are {cls.EXPECTED_INPUT_DIMENSIONS}.")

        if output_dimension not in cls.EXPECTED_OUTPUT_DIMENSIONS:
            raise ValueError(f"The output dimension of the given vector field is not correct. Current dimension is "
                             f"{output_dimension} while accepted dimensions are {cls.EXPECTED_OUTPUT_DIMENSIONS}.")

        return np.asarray(deepcopy(field)).view(cls)

    @property
    def input_dimension(self) -> int:
        """
        The map's input dimension.

                 ┏━ The dimension of this set.
                 ┃
            f : ℝⁿ → ℝ².
        """
        return len(self.shape[:-1])

    @property
    def output_dimension(self) -> int:
        """
        The map's output dimension.

                     ┏━ The dimension of this set.
                     ┃
            f : ℝⁿ → ℝ².
        """
        return self.shape[-1]

    @property
    def x(self) -> ScalarField:
        """
        Scalar field f : ℝ² → ℝ ; (x, y) → F_x.
        """
        return ScalarField(self[..., 0])

    @property
    def y(self) -> ScalarField:
        """
        Scalar field f : ℝ² → ℝ ; (x, y) → F_y.
        """
        return ScalarField(self[..., 1])

    @property
    def z(self) -> Optional[ScalarField]:
        """
        Scalar field f : ℝ² → ℝ ; (x, y) → F_z.
        """
        if self.output_dimension == 2:
            return None
        elif self.output_dimension == 3:
            return ScalarField(self[..., 2])

    def cross(self, field: VectorField) -> VectorField:
        """
        Cross product of 2 vector fields.

        Parameters
        ----------
        field : VectorField
            A vector field.

        Returns
        -------
        field : VectorField
            The cross product between the current vector field and the given one.
        """
        return VectorField(np.cross(self, field))

    def __stream_plot_color_bar(self) -> plt.streamplot:
        """
        Creates a temporary stream plot used for the colorbar of the field's figure.

        Returns
        -------
        temp_stream_plot : plt.streamplot
            A temporary stream plot used for the colorbar of the field's figure.
        """
        temp_fig = plt.figure()
        temp_ax = temp_fig.add_subplot(111)
        temp_stream_plot = temp_ax.streamplot(
            x=np.arange(0, self.shape[0]),
            y=np.arange(0, self.shape[1]),
            u=self.x.T,
            v=self.y.T,
            color=np.hypot(self.x.T, self.y.T),
            linewidth=1,
            cmap=plt.cm.inferno,
            density=3,
            arrowstyle='->',
            arrowsize=1.5
        )

        plt.close(temp_fig)

        return temp_stream_plot

    def show(self, **kwargs):
        """
        Show the vector field in the xy plane.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments to create a custom matplotlib figure. Keyword arguments currently defined are :
                {
                x_label : str
                    Label text for the x-axis.
                y_label : str
                    Label text for the y-axis.
                title : str
                    Text to use for the figure's title.
            }
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        with np.errstate(divide='ignore'):
            color = 2 * np.log(np.hypot(self.x.T, self.y.T))

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)

            stream_plot = ax.streamplot(
                x=np.arange(0, self.shape[0]),
                y=np.arange(0, self.shape[1]),
                u=self.x.T,
                v=self.y.T,
                color=color,
                linewidth=1,
                cmap=plt.cm.inferno,
                density=3,
                arrowstyle='->',
                arrowsize=1.5
            )

        ax.set_xlabel(xlabel=kwargs.get("x_label", "x [m]"))
        ax.set_ylabel(ylabel=kwargs.get("y_label", "y [m]"))
        ax.set_title(label=kwargs.get("title", ""))
        fig.colorbar(self.__stream_plot_color_bar().lines, orientation='vertical')
        plt.show()
