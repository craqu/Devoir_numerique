import env_tests  # Modifies path, DO NOT REMOVE
import unittest

from src.circuit import Circuit
from src.wire import Current, Wire
from src.world import World


class TestWorld(unittest.TestCase):

    WORLD_SHAPE = (51, 51)
    CENTER = (int((WORLD_SHAPE[0] - 1)/2), int((WORLD_SHAPE[1] - 1)/2))

    WIRES = [
        Wire(start=(13, 25), stop=(13, 37), current=Current(x=0, y=1), voltage=4.5),
        Wire(start=(13, 37), stop=(37, 37), current=Current(x=1, y=0), voltage=4.5),
        Wire(start=(37, 37), stop=(37, 25), current=Current(x=0, y=-1), voltage=4.5),
        Wire(start=(37, 25), stop=(37, 13), current=Current(x=0, y=-1), voltage=-4.5),
        Wire(start=(37, 13), stop=(13, 13), current=Current(x=-1, y=0), voltage=-4.5),
        Wire(start=(13, 13), stop=(13, 25), current=Current(x=0, y=1), voltage=-4.5),
    ]

    CIRCUIT = Circuit(wires=WIRES)

    def setUp(self):
        self.world = World(self.WORLD_SHAPE)
        self.world.place(self.CIRCUIT)
        self.world.compute()
    def testShouldReturnTheRightMagneticVectorAtCenter(self):
        true_value = [0, 0, -4.71199948e-08]
        computed_value = self.world._magnetic_field[self.CENTER]

        self.assertAlmostEqual(computed_value[0], true_value[0])
        self.assertAlmostEqual(computed_value[1], true_value[1])
        self.assertAlmostEqual(computed_value[2], true_value[2])

    def testShouldReturnTheRightPotentialAtCenter(self):
        true_value = -0.1573155955205982
        computed_value = self.world._potential[self.CENTER]

        self.assertAlmostEqual(computed_value, true_value)

    def testShouldReturnTheRightElectricVectorAtCenter(self):
        true_value = [-7.11430498e-11, -4.42911963e-01, 0]
        computed_value = self.world._electric_field[self.CENTER]

        self.assertAlmostEqual(computed_value[0], true_value[0])
        self.assertAlmostEqual(computed_value[1], true_value[1])
        # We deliberately don't test the last value of the vector, because it's equal to 0 and could therefore be
        # omitted.

    def testShouldReturnTheRightEnergyFluxVectorAtCenter(self):
        true_value = [1.66078258e-02, -2.66764386e-12, 0]
        computed_value = self.world._energy_flux[self.CENTER]

        self.assertAlmostEqual(computed_value[0], true_value[0])
        self.assertAlmostEqual(computed_value[1], true_value[1])
        # We deliberately don't test the last value of the vector, because it's equal to 0 and could therefore be
        # omitted.
if __name__ == "__main__":
    unittest.main()
