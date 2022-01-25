# An Invertible Method For Material Characterization via the Inverse Material Point Method
from femflow.reconstruction.behavior_matching import BehaviorMatching
from femflow.simulation.mpm.gui import MPMDisplacementsWindow, MPMSimulationWindow
from femflow.simulation.mpm.primitives import generate_implicit_points, gyroid
from femflow.simulation.mpm.simulation import MPMSimulation
from femflow.viz.mesh import Mesh
from femflow.viz.visualizer.window import Window


def load_view():
    with Window("paper 1") as window:
        sim = MPMSimulation()
        window.add_window(
            MPMSimulationWindow(),
            sim_status=sim.loaded,
            load_sim_cb=sim.load,
            start_sim_button_cb=sim.start,
            mesh=window.renderer.mesh,
        )
        window.add_window(MPMDisplacementsWindow(), sim=sim)
        window.launch()


def run_experiment():
    print("Hey there")
