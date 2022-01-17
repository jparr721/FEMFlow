class Parameters(object):
    def __init__(
        self,
        mass: float,
        volume: float,
        hardening: float,
        E: float,
        nu: float,
        gravity: float,
        dt: float,
        grid_resolution: int,
    ):
        self.mass = mass
        self.volume = volume
        self.hardening = hardening

        self.E = E
        self.nu = nu

        self.mu_0 = E / (2 * (1 + nu))
        self.lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

        self.gravity = gravity
        self.dt = dt
        self.grid_resolution = grid_resolution

        # dx is always 1 / grid_resolution
        self.dx = 1 / grid_resolution
        self.inv_dx = 1 / self.dx
