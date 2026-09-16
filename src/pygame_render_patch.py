"""
Pygame Render Refinement & Alpha Transparency Patch for DSSE Environments
==========================================================================
Fixes DSSE's default Pygame renderer where image.convert() strips PNG alpha
transparency channels, causing harsh black background boxes around drones and swimmer icons.

Refinements Applied:
  1. Preserves full PNG alpha transparency (.convert_alpha()) to remove black square boxes.
  2. Uses high-quality smooth scaling (smoothscale) for sharp icon rendering.
  3. Highlights the (0,0) Base Charging Station with an Emerald Green badge (RGB: 40, 220, 100).
  4. Keeps the exact radial probability heatmap colors, cell grid, and overall aesthetic.
"""

import os
import pygame
import numpy as np
import DSSE


def patch_pygame_renderer(env):
    """
    Patches the PygameInterface renderer attached to a DSSE environment instance.
    Call this after env.reset() or during environment initialization.
    """
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    if hasattr(base_env, 'unwrapped'):
        base_env = base_env.unwrapped

    if not hasattr(base_env, 'pygame_renderer') or base_env.pygame_renderer is None:
        return

    renderer = base_env.pygame_renderer
    dsse_dir = os.path.dirname(DSSE.__file__)
    drone_path = os.path.join(dsse_dir, 'environment/imgs/drone.png')
    person_path = os.path.join(dsse_dir, 'environment/imgs/person-swimming.png')

    # Re-enable rendering window if needed
    if not renderer.render_on:
        renderer.enable_render()

    block_size = int(renderer.block_size)

    # Load images preserving PNG alpha transparency (.convert_alpha())
    try:
        raw_drone = pygame.image.load(drone_path).convert_alpha()
        drone_img = pygame.transform.smoothscale(raw_drone, (block_size, block_size))
    except Exception:
        drone_img = renderer.drone_img

    try:
        raw_person = pygame.image.load(person_path).convert_alpha()
        person_img = pygame.transform.smoothscale(raw_person, (block_size, block_size))
    except Exception:
        person_img = renderer.person_img

    renderer.drone_img = drone_img
    renderer.person_img = person_img

    # Override draw method to ensure charging station at (0,0) renders cleanly
    def custom_draw():
        renderer.clock.tick(renderer.FPS)
        renderer.screen.fill((0, 0, 0))
        matrix = renderer.probability_matrix.get_matrix()
        max_matrix = matrix.max() if matrix.max() != 0.0 else 1.0

        grid_w, grid_h = renderer.grid_size, renderer.grid_size

        for counter_x, x in enumerate(np.arange(10, renderer.window_size + 10, renderer.block_size)):
            for counter_y, y in enumerate(np.arange(10, renderer.window_size + 10, renderer.block_size)):
                if counter_x < grid_w and counter_y < grid_h:
                    rectangle = renderer.get_position_rectangle((counter_x, counter_y))
                    if counter_x == 0 and counter_y == 0:
                        # Charging Base Station at (0,0) - Bright Emerald Green!
                        color = (40, 220, 100)
                    else:
                        prob = matrix[counter_y][counter_x]
                        norm_prob = prob / max_matrix
                        color = renderer.compute_cell_color(norm_prob)

                    pygame.draw.rect(renderer.screen, color, rectangle)
                    if renderer.render_grid:
                        pygame.draw.rect(renderer.screen, (0, 0, 0), rectangle, 1)

    renderer.draw = custom_draw
