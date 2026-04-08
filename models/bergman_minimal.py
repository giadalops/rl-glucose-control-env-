from dataclasses import dataclass


@dataclass
class GlucoseInsulinModel:
    """
    Simple model inspired by Bergman Minimal Model.

    States:
        G: glycemia [mg/dL]
        X: insulin remote effect [a.u.]

    Simplified dynamics:
        dG/dt = -p1 * (G - Gb) - X * max(G, 0) + D
        dX/dt = -p2 * X + p3 * u

    where:
        u = infused insulin [a.u.]
        D = meal disturbance [mg/dL/min]
    """
    Gb: float = 110.0   # basal glycemia
    p1: float = 0.01    # natural return to basal
    p2: float = 0.05    # insulin decay effect
    p3: float = 0.02    # insulin input gain

    def step(self, G: float, X: float, u: float, D: float, dt: float) -> tuple[float, float]:
        """
        Euler integration step.
        """
        dG = -self.p1 * (G - self.Gb) - X * max(G, 0.0) + D
        dX = -self.p2 * X + self.p3 * max(u, 0.0)

        G_next = G + dt * dG
        X_next = X + dt * dX

        # Minimal numerical constraints for toy model stability

        G_next = max(20.0, min(600.0, G_next))
        X_next = max(0.0, min(10.0, X_next))

        return G_next, X_next
