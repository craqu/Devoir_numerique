import os
import sys

# Append module root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.circuit import Circuit
from src.wire import Current, Wire
from src.world import World


WORLD_SHAPE = (51, 51)

WIRES = [
    Wire(start=(13, 25), stop=(13, 37), current=Current(x=0, y=1), voltage=4.5),
    Wire(start=(13, 37), stop=(37, 37), current=Current(x=1, y=0), voltage=4.5),
    Wire(start=(37, 37), stop=(37, 25), current=Current(x=0, y=-1), voltage=4.5),
    Wire(start=(37, 25), stop=(37, 13), current=Current(x=0, y=-1), voltage=-4.5),
    Wire(start=(37, 13), stop=(13, 13), current=Current(x=-1, y=0), voltage=-4.5),
    Wire(start=(13, 13), stop=(13, 25), current=Current(x=0, y=1), voltage=-4.5),
]
CIRCUIT = Circuit(wires=WIRES)

world = World(WORLD_SHAPE)
world.place(CIRCUIT)
world.compute()
world.show_all()