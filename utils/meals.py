from dataclasses import dataclass, field

@dataclass
class MealScenario:
    """
    A simple simulation of dietary intake disturbances.

    Each meal is defined by:
        {"time": start_minute, "size": intensity}

    The generated disturbance persists for a fixed window and then decays.
    """
    meals: list[dict] = field(default_factory=list)
    meal_duration: int = 60  # Duration in minutes for the "effective" impact

    @classmethod
    def default(cls) -> "MealScenario":
        """Returns a standard daily scenario with breakfast, lunch, and dinner."""
        return cls(
            meals=[
                {"time": 120, "size": 18.0},  # Breakfast
                {"time": 480, "size": 25.0},  # Lunch
                {"time": 780, "size": 22.0},  # Dinner
            ],
            meal_duration=60,
        )

    def disturbance(self, t: int) -> float:
        """
        Calculates the total meal disturbance at time 't'.
        Model: Linear decay within the meal_duration window.
        """
        total = 0.0
        for meal in self.meals:
            start = meal["time"]
            size = meal["size"]
            
            # Check if the current time falls within the meal impact window
            if start <= t < start + self.meal_duration:
                # Calculate the remaining linear fraction (from 1.0 down to 0.0)
                decay_fraction = 1.0 - (t - start) / self.meal_duration
                # Add the normalized intensity to the total disturbance
                total += (size * decay_fraction) / self.meal_duration
                
        return total
