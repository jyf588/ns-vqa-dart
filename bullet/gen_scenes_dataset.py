"""
Generates a dataset containing various scenes of random objects placed randomly
on a tabletop, all upright. Currently does not generate examples of stacked 
objects. Robot arm is not in view for this dataset.
"""
import time

from bullet.scenes import RandomSceneGenerator


def main():
    generator = RandomSceneGenerator(
        seed=1,
        render_mode="gui",
        egocentric=True,
        n_objs_bounds=(3, 7),
        shapes=["box", "cylinder"],
        sizes=["large", "small"],
        colors=["red", "blue", "yellow", "green"],
        x_bounds=(0.0, 0.4),
        y_bounds=(-0.5, 0.5),
        z_bounds=(0.0, 0.0),
        roll_bounds=(-5.0, 5.0),
        tilt_bounds=(40.0, 60.0),
        pan_bounds=(0.0, 0.0),
        degrees=True,
    )
    generator.generate_scenes(n=22000)


if __name__ == "__main__":
    main()
