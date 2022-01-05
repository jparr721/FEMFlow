from .linear_fem_simulation import LinearFemSimulationHeadless
from femflow.viz.mesh import Mesh
import numpy as np
import copy
from jax import grad
from tqdm import tqdm
from loguru import logger


class GalerkinOptimizer(object):
    def __init__(
        self, mesh: Mesh, force: float, target_U: float, epochs=300, learning_rate=0.01
    ):
        self.name = "galerkin_optimizer"

        # Somewhere between here and solid steel
        # Youngs modulus represents our "weight" value
        # self.youngs_modulus = np.random.randint(0, 210e6)
        self.youngs_modulus = 50000
        self.poissons_ratio = 0.3

        self.mesh = mesh
        self.force = force
        self.target_U = target_U
        self.epochs = epochs
        self.learning_rate = learning_rate

    def predict(self) -> float:
        mesh_clone = copy.deepcopy(self.mesh)

        simulation = LinearFemSimulationHeadless(
            force=self.force,
            youngs_modulus=self.youngs_modulus,
            poissons_ratio=self.poissons_ratio,
        )
        simulation.load(mesh_clone)
        start_top_nodes = list(
            map(
                lambda x: x[1],
                mesh_clone.as_matrix(mesh_clone.vertices, 3)[simulation.force_nodes],
            )
        )
        start_top = np.average(start_top_nodes)
        bottom_nodes = list(
            map(
                lambda x: x[1],
                mesh_clone.as_matrix(mesh_clone.vertices, 3)[simulation.fixed_nodes],
            )
        )
        bottom = np.average(bottom_nodes)

        # Initial height
        initial_height = start_top - bottom

        simulation.solve_static()
        mesh_clone.transform(simulation.solver.U)
        # Average the force-applied nodes (top)
        end_top_nodes = list(
            map(
                lambda x: x[1],
                mesh_clone.as_matrix(mesh_clone.vertices, 3)[simulation.force_nodes],
            )
        )
        end_top = np.average(end_top_nodes)

        # Bottom should not have changed
        end_height = end_top - bottom

        # We want the proportional difference
        return 100 - ((end_height / initial_height) * 100)

    def train(self):
        initial_prediction = self.predict()
        initial_loss = self.loss(initial_prediction)
        logger.info(f"Intial loss: {initial_loss}")

        gradient_loss_fn = grad(self.loss)

        progressbar = tqdm(range(self.epochs))
        for _ in tqdm(range(self.epochs)):
            displacement = self.predict()
            progressbar.set_postfix(
                {
                    "loss": self.loss(displacement),
                    "E": self.youngs_modulus,
                    "displacement": displacement,
                    "target": self.target_U,
                }
            )
            self.youngs_modulus += gradient_loss_fn(displacement) * self.learning_rate

        logger.success(f"Optimum value of E has been saved.")

    def loss(self, sample: float) -> np.float32:
        """Standard mean squared error loss function. This function
        currently operates only on floating point input, vector-based
        inputs will be added later.

        Args:
            sample (float): The sample from the prediction.

        Returns:
            np.float32: The mean squared error loss
        """
        return np.mean((sample - self.target_U) ** 2)
