import pygame
import math
import sys
import numpy as np
import random
import matplotlib.cm as cm

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (64, 64, 64)
LIGHT_GRAY = (128, 128, 128)
DARK_GRAY = (32, 32, 32)

# Calculate normalized coordinate system based on screen aspect ratio
SCREEN_ASPECT = SCREEN_WIDTH / SCREEN_HEIGHT
BASE_HEIGHT = 4  # Fixed height for consistent scale
NORM_HEIGHT = BASE_HEIGHT
NORM_WIDTH = BASE_HEIGHT * SCREEN_ASPECT  # Adjust width to match screen aspect ratio
NORM_CENTER_X = 0
NORM_CENTER_Y = 0

def norm_to_screen(x_norm, y_norm):
    """Convert normalized coordinates to screen pixels."""
    x_screen = (x_norm - NORM_CENTER_X + NORM_WIDTH/2) / NORM_WIDTH * SCREEN_WIDTH
    y_screen = SCREEN_HEIGHT - (y_norm - NORM_CENTER_Y + NORM_HEIGHT/2) / NORM_HEIGHT * SCREEN_HEIGHT
    # Y flipped to have positive Y upwards in normalized coordinates
    return x_screen, y_screen

def screen_to_norm(x_screen, y_screen):
    """Convert screen pixels to normalized coordinates."""
    x_norm = (x_screen / SCREEN_WIDTH) * NORM_WIDTH - NORM_WIDTH/2 + NORM_CENTER_X
    y_norm = (1 - y_screen / SCREEN_HEIGHT) * NORM_HEIGHT - NORM_HEIGHT/2 + NORM_CENTER_Y
    return x_norm, y_norm

def is_contractive(a, b, c, d, e, f):
    """Check if the affine transformation parameters form a contractive mapping."""
    condition1 = a*a + d*d < 1
    condition2 = b*b + e*e < 1
    condition3 = a*a + b*b + d*d + e*e < 1 + (a*e - d*b)**2
    return condition1 and condition2 and condition3

def decompose_affine(a, b, c, d, e, f):
    """
    Decompose 2D affine transformation:
        x' = a*x + b*y + c
        y' = d*x + e*y + f
    into scaling (sx, sy), rotation theta, and translation (px, py)
    
    Returns:
        sx, sy, theta, px, py
    """
    # Translation
    px = c
    py = f
    
    # Compute scaling factors
    sx = math.sqrt(a*a + d*d)
    sy = math.sqrt(b*b + e*e)
    
    # Compute rotation angle
    # Note: assumes no shear, no reflection
    theta = math.atan2(d, a)
    
    return sx, sy, theta, px, py

def generate_random_contractive_params():
    """Generate random contractive affine transformation parameters."""
    max_attempts = 1000
    for _ in range(max_attempts):
        # Generate random parameters in range [-1, 1]
        a = random.uniform(-1, 1)
        b = random.uniform(-1, 1)
        c = random.uniform(-2, 2)
        d = random.uniform(-1, 1)
        e = random.uniform(-2, 2)
        f = random.uniform(-1, 1)
        
        if is_contractive(a, b, c, d, e, f):
            return a, b, c, d, e, f
    
    # Fallback to a known contractive transformation if we can't generate one
    return 0.5, 0, 0, 0, 0.5, 0

class Rectangle:
    def __init__(self, scale_x=0.5, scale_y=0.5, theta=0, center_x=0, center_y=0):
        """
        Rectangle in normalized coordinate system:
        scale_x, scale_y: half-width and half-height in normalized units
        theta: rotation in radians
        center_x, center_y: position in normalized coordinates
        """
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.theta = theta
        self.center_x = center_x
        self.center_y = center_y
        self.selected = False

    @property
    def transformation_matrix(self):
        """Combined transformation matrix M = T * R * S in normalized coords."""
        S = np.array([
            [self.scale_x, 0, 0],
            [0, self.scale_y, 0],
            [0, 0, 1]
        ])
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        R = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        T = np.array([
            [1, 0, self.center_x],
            [0, 1, self.center_y],
            [0, 0, 1]
        ])
        return T @ R @ S

    @property
    def rigid_motion_matrix(self):
        """Rigid motion matrix M = T * R without scaling."""
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        R = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        T = np.array([
            [1, 0, self.center_x],
            [0, 1, self.center_y],
            [0, 0, 1]
        ])
        return T @ R

    def get_corners(self):
        """Get corners in normalized coords."""
        unit_corners = np.array([
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ]).T
        transformed = self.transformation_matrix @ unit_corners
        corners = []
        for i in range(4):
            corners.append((transformed[0, i], transformed[1, i]))
        return corners

    def get_edge_midpoints(self):
        corners = self.get_corners()
        midpoints = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            midpoints.append((mid_x, mid_y))
        return midpoints

    def get_right_direction(self):
        right_local = np.array([1, 0, 0])
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        R = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        right_world = R @ right_local
        return (right_world[0], right_world[1])

    def draw(self, screen, show_ui=True):
        """Draw rectangle with scaling to screen pixels."""
        corners_norm = self.get_corners()
        corners_screen = [norm_to_screen(x, y) for x, y in corners_norm]

        # Draw rectangle outline
        pygame.draw.polygon(screen, WHITE, corners_screen, 2)

        if not show_ui:
            return

        # Orientation arrow
        right_dir = self.get_right_direction()
        arrow_length_norm = 0.75  # normalized length for arrow
        arrow_end_x_norm = self.center_x + right_dir[0] * arrow_length_norm
        arrow_end_y_norm = self.center_y + right_dir[1] * arrow_length_norm
        start_screen = norm_to_screen(self.center_x, self.center_y)
        end_screen = norm_to_screen(arrow_end_x_norm, arrow_end_y_norm)

        pygame.draw.line(screen, WHITE, start_screen, end_screen, 3)

        # Arrowhead calculation in screen space (small fixed size)
        head_size = 8
        dir_angle = math.atan2(end_screen[1] - start_screen[1], end_screen[0] - start_screen[0])
        head_angle1 = dir_angle + math.pi - 0.5
        head_angle2 = dir_angle + math.pi + 0.5

        head_x1 = end_screen[0] + head_size * math.cos(head_angle1)
        head_y1 = end_screen[1] + head_size * math.sin(head_angle1)
        head_x2 = end_screen[0] + head_size * math.cos(head_angle2)
        head_y2 = end_screen[1] + head_size * math.sin(head_angle2)

        pygame.draw.polygon(screen, WHITE,
                            [(end_screen[0], end_screen[1]), (head_x1, head_y1), (head_x2, head_y2)])

        if self.selected:
            # Draw movement arrows around center in screen coords
            arrow_size = 12
            center_screen_x, center_screen_y = start_screen

            # Up arrow
            pygame.draw.polygon(screen, RED, [
                (center_screen_x, center_screen_y - arrow_size),
                (center_screen_x - 4, center_screen_y - arrow_size + 6),
                (center_screen_x + 4, center_screen_y - arrow_size + 6)
            ])
            # Down arrow
            pygame.draw.polygon(screen, RED, [
                (center_screen_x, center_screen_y + arrow_size),
                (center_screen_x - 4, center_screen_y + arrow_size - 6),
                (center_screen_x + 4, center_screen_y + arrow_size - 6)
            ])
            # Left arrow
            pygame.draw.polygon(screen, RED, [
                (center_screen_x - arrow_size, center_screen_y),
                (center_screen_x - arrow_size + 6, center_screen_y - 4),
                (center_screen_x - arrow_size + 6, center_screen_y + 4)
            ])
            # Right arrow
            pygame.draw.polygon(screen, RED, [
                (center_screen_x + arrow_size, center_screen_y),
                (center_screen_x + arrow_size - 6, center_screen_y - 4),
                (center_screen_x + arrow_size - 6, center_screen_y + 4)
            ])

            # Draw rotation handle - circle in screen space
            # Calculate radius based on rectangle size in screen space
            radius_norm = max(abs(self.scale_x), abs(self.scale_y)) * 1.2 + 0.3
            # Convert radius from normalized space to screen pixels
            center_screen_norm_x = self.center_x + radius_norm
            center_screen_norm_y = self.center_y
            radius_point_screen = norm_to_screen(center_screen_norm_x, center_screen_norm_y)
            radius_pix = abs(radius_point_screen[0] - center_screen_x)

            pygame.draw.circle(screen, GRAY, (int(center_screen_x), int(center_screen_y)), int(radius_pix), 1)

            # Circular arrow indicators in screen coords
            for angle in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
                start_x = center_screen_x + radius_pix * math.cos(angle)
                start_y = center_screen_y + radius_pix * math.sin(angle)
                end_angle = angle + 0.3
                end_x = center_screen_x + radius_pix * math.cos(end_angle)
                end_y = center_screen_y + radius_pix * math.sin(end_angle)

                pygame.draw.line(screen, GRAY, (start_x, start_y), (end_x, end_y), 2)

                arrow_head_angle = end_angle + math.pi / 2
                head_size = 4
                head_x1 = end_x + head_size * math.cos(arrow_head_angle + 0.5)
                head_y1 = end_y + head_size * math.sin(arrow_head_angle + 0.5)
                head_x2 = end_x + head_size * math.cos(arrow_head_angle - 0.5)
                head_y2 = end_y + head_size * math.sin(arrow_head_angle - 0.5)
                pygame.draw.polygon(screen, GRAY, [(end_x, end_y), (head_x1, head_y1), (head_x2, head_y2)])

            # Draw corner handles (scaled to screen coords)
            for corner_norm in corners_norm:
                corner_screen = norm_to_screen(*corner_norm)
                pygame.draw.rect(screen, WHITE,
                                 (corner_screen[0] - 4, corner_screen[1] - 4, 8, 8))
                pygame.draw.rect(screen, BLUE,
                                 (corner_screen[0] - 4, corner_screen[1] - 4, 8, 8), 2)

            # Draw edge midpoint handles
            for midpoint_norm in self.get_edge_midpoints():
                midpoint_screen = norm_to_screen(*midpoint_norm)
                pygame.draw.rect(screen, WHITE,
                                 (midpoint_screen[0] - 3, midpoint_screen[1] - 3, 6, 6))
                pygame.draw.rect(screen, GREEN,
                                 (midpoint_screen[0] - 3, midpoint_screen[1] - 3, 6, 6), 2)

    def contains_point(self, px, py):
        """Check if a screen pixel point is inside the rectangle by inverse transforming to normalized space."""
        try:
            M = self.transformation_matrix
            M_inv = np.linalg.inv(M)
            px_norm, py_norm = screen_to_norm(px, py)
            point_world = np.array([px_norm, py_norm, 1])
            point_local = M_inv @ point_world
            return (-1 <= point_local[0] <= 1) and (-1 <= point_local[1] <= 1)
        except np.linalg.LinAlgError:
            return False

    def get_interaction_type(self, px, py):
        """Determine interaction type from screen pixel position."""
        if not self.selected:
            return None

        # Convert corners and midpoints to screen coordinates for proximity checking
        corners = self.get_corners()
        corners_screen = [norm_to_screen(x, y) for x, y in corners]
        for i, corner in enumerate(corners_screen):
            dist = math.sqrt((px - corner[0]) ** 2 + (py - corner[1]) ** 2)
            if dist <= 10:
                return f"scale_corner_{i}"

        midpoints = self.get_edge_midpoints()
        midpoints_screen = [norm_to_screen(x, y) for x, y in midpoints]
        for i, midpoint in enumerate(midpoints_screen):
            dist = math.sqrt((px - midpoint[0]) ** 2 + (py - midpoint[1]) ** 2)
            if dist <= 8:
                return f"scale_edge_{i}"

        center_screen = norm_to_screen(self.center_x, self.center_y)
        center_dist = math.sqrt((px - center_screen[0]) ** 2 + (py - center_screen[1]) ** 2)
        if center_dist <= 18:
            return "move"

        # Calculate rotation handle radius in screen space
        radius_norm = max(abs(self.scale_x), abs(self.scale_y)) * 1.2 + 0.3
        center_screen_norm_x = self.center_x + radius_norm
        center_screen_norm_y = self.center_y
        radius_point_screen = norm_to_screen(center_screen_norm_x, center_screen_norm_y)
        radius_pix = abs(radius_point_screen[0] - center_screen[0])

        if radius_pix >= center_dist > 25:
            return "rotate"

        return None


class RectangleEditor:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("2D Rectangle Editor with Proper Aspect Ratio")

        self.clock = pygame.time.Clock()

        # Initialize rectangles with reasonable normalized coordinates
        self.rectangles = []
        self.selected_rect = None
        self.interaction_mode = None
        self.last_mouse_pos = None
        self.initial_angle = 0
        self.initial_scale = (0, 0)

        
        # Always show both fractal and rectangles together
        self.fractal_iterations = 50000
        self.editing_iterations = 20000  # Lower iterations during editing
        self.fractal_counts = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.int32)  # Count matrix
        self.fractal_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.editing_iterations_ui = False
        self.iteration_input = ""
        
        # Fractal state for continuous rendering
        self.current_point = np.array([0, 0, 1])  # Current point for continuous iteration
        self.is_editing = False
        self.colormap_name = "grey"  # Default colormap
        
        # UI visibility toggle
        self.show_ui = True

        self.generate_random_ifs()
        
        # Generate initial fractal
        self.generate_fractal()

    def generate_random_ifs(self):
        """Generate 3-5 rectangles from random contractive IFS parameters."""
        self.rectangles.clear()
        self.selected_rect = None
        
        num_rectangles = random.randint(3, 6)
        
        for _ in range(num_rectangles):
            # Generate random contractive parameters
            a, b, c, d, e, f = generate_random_contractive_params()
            
            # Decompose into scale, rotation, and translation
            sx, sy, theta, px, py = decompose_affine(a, b, c, d, e, f)
            
            # Create rectangle
            rect = Rectangle(sx, sy, theta, px, py)
            self.rectangles.append(rect)
        
        # Generate fractal immediately
        if self.rectangles:
            self.generate_fractal(force_full=True)

    def update_fractal_surface(self):
        """Convert the count matrix to a colormap surface."""
        # Normalize counts to [0, 1] distribution
        sum_count = np.sum(self.fractal_counts)
        if sum_count > 0: 
            distribution = np.log1p(self.fractal_counts)
            distribution = 1 / (1 + np.exp(-distribution)) * 2 - 1
            distribution = np.clip(distribution, 0, 1)
        else:
            distribution = self.fractal_counts.astype(np.float64)
        
        # Apply colormap to get RGB image
        rgb_img = self.apply_colormap(distribution, self.colormap_name)
        
        # Convert to 8-bit RGB and create surface
        rgb_array = (rgb_img * 255).astype(np.uint8)
        self.fractal_surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))

    def continue_fractal_rendering(self):
        """Continue adding points to the fractal when not editing."""
        if len(self.rectangles) == 0 or self.is_editing:
            return

        matrices = []
        for rect in self.rectangles:
            matrices.append(rect.transformation_matrix)

        if not matrices:
            return

        # Add more points to the existing fractal
        p = self.current_point.copy()
        iterations_per_frame = 1000  # Add this many points per frame
        
        for _ in range(iterations_per_frame):
            # Convert to screen pixels for plotting
            x_pix, y_pix = norm_to_screen(p[0], p[1])
            
            # Check bounds and increment count
            if 0 <= x_pix < SCREEN_WIDTH and 0 <= y_pix < SCREEN_HEIGHT:
                self.fractal_counts[int(y_pix), int(x_pix)] += 1

            # Apply random transformation
            M = random.choice(matrices)
            p = M @ p

        # Update current point
        self.current_point = p
        
        # Update surface
        self.update_fractal_surface()

    def apply_colormap(self, data, cmap_name="hot"):
        """
        Map a 2D array of values in [0,1] to RGB using a colormap.
        
        Parameters:
            data (np.ndarray): 2D array, values in [0,1]
            cmap_name (str): matplotlib colormap name ("viridis", "plasma", "inferno", "magma", "hot", etc.)
        
        Returns:
            np.ndarray: 3D array (H, W, 3) with RGB values in [0,1]
        """
        cmap = cm.get_cmap(cmap_name)
        rgba_img = cmap(data)           # shape (H, W, 4), includes alpha
        rgb_img = (rgba_img[..., :3])   # drop alpha
        return rgb_img

    def generate_fractal(self, force_full=False):
        if len(self.rectangles) == 0:
            return

        matrices = []
        for rect in self.rectangles:
            matrices.append(rect.transformation_matrix)

        if not matrices:
            return

        # Determine iteration count based on editing state
        if self.is_editing and not force_full:
            iterations = self.editing_iterations
            # Clear and restart for editing
            self.fractal_counts.fill(0)
            self.current_point = np.array([0, 0, 1])
        else:
            iterations = self.fractal_iterations
            if force_full:
                # Clear and restart for full generation
                self.fractal_counts.fill(0)
                self.current_point = np.array([0, 0, 1])

        # Generate fractal points
        p = self.current_point.copy()
        for _ in range(iterations):
            # Convert to screen pixels for plotting
            x_pix, y_pix = norm_to_screen(p[0], p[1])
            
            # Check bounds and increment count
            if 0 <= x_pix < SCREEN_WIDTH and 0 <= y_pix < SCREEN_HEIGHT:
                self.fractal_counts[int(y_pix), int(x_pix)] += 1

            # Apply random transformation
            M = random.choice(matrices)
            p = M @ p

        # Update current point for continuous rendering
        self.current_point = p

        # Create surface from count matrix
        self.update_fractal_surface()

    def draw_grid(self):
        """Draw grid lines in normalized coordinate system with improved axis marking."""
        if not self.show_ui:
            return
            
        # Grid spacing in normalized coordinates
        grid_spacing_x = 1.0
        grid_spacing_y = 1.0

        # Draw vertical grid lines
        x = int(-NORM_WIDTH/2)
        while x <= int(NORM_WIDTH/2):
            x_screen_top, y_screen_top = norm_to_screen(x, NORM_HEIGHT/2)
            x_screen_bottom, y_screen_bottom = norm_to_screen(x, -NORM_HEIGHT/2)
            if x == 0:
                # Main Y-axis - more prominent
                pygame.draw.line(self.screen, LIGHT_GRAY, (x_screen_top, y_screen_top), (x_screen_bottom, y_screen_bottom), 3)
            else:
                pygame.draw.line(self.screen, DARK_GRAY, (x_screen_top, y_screen_top), (x_screen_bottom, y_screen_bottom), 1)
            x += grid_spacing_x

        # Draw horizontal grid lines
        y = int(-NORM_HEIGHT/2)
        while y <= int(NORM_HEIGHT/2):
            x_screen_left, y_screen_left = norm_to_screen(-NORM_WIDTH/2, y)
            x_screen_right, y_screen_right = norm_to_screen(NORM_WIDTH/2, y)
            if y == 0:
                # Main X-axis - more prominent
                pygame.draw.line(self.screen, LIGHT_GRAY, (x_screen_left, y_screen_left), (x_screen_right, y_screen_right), 3)
            else:
                pygame.draw.line(self.screen, DARK_GRAY, (x_screen_left, y_screen_left), (x_screen_right, y_screen_right), 1)
            y += grid_spacing_y

        # Draw axis tick marks and labels
        font = pygame.font.Font(None, 20)
        
        # X-axis tick marks and labels
        for x_val in range(int(-NORM_WIDTH/2), int(NORM_WIDTH/2) + 1):
            if -NORM_WIDTH/2 <= x_val <= NORM_WIDTH/2:
                x_pos_screen = norm_to_screen(x_val, 0)
                
                # Draw tick mark
                tick_length = 8
                pygame.draw.line(self.screen, WHITE, 
                               (x_pos_screen[0], x_pos_screen[1] - tick_length//2), 
                               (x_pos_screen[0], x_pos_screen[1] + tick_length//2), 2)
                
                # Draw label (skip 0 to avoid overlap)
                if x_val != 0:
                    x_text = font.render(str(x_val), True, WHITE)
                    text_rect = x_text.get_rect()
                    self.screen.blit(x_text, (x_pos_screen[0] - text_rect.width//2, x_pos_screen[1] + 15))
        
        # Y-axis tick marks and labels
        for y_val in range(int(-NORM_HEIGHT/2), int(NORM_HEIGHT/2) + 1):
            if -NORM_HEIGHT/2 <= y_val <= NORM_HEIGHT/2:
                y_pos_screen = norm_to_screen(0, y_val)
                
                # Draw tick mark
                tick_length = 8
                pygame.draw.line(self.screen, WHITE,
                               (y_pos_screen[0] - tick_length//2, y_pos_screen[1]),
                               (y_pos_screen[0] + tick_length//2, y_pos_screen[1]), 2)
                
                # Draw label (skip 0 to avoid overlap)
                if y_val != 0:
                    y_text = font.render(str(y_val), True, WHITE)
                    text_rect = y_text.get_rect()
                    self.screen.blit(y_text, (y_pos_screen[0] + 15, y_pos_screen[1] - text_rect.height//2))
        
        # Origin label
        origin_screen = norm_to_screen(0, 0)
        origin_text = font.render("0", True, WHITE)
        text_rect = origin_text.get_rect()
        self.screen.blit(origin_text, (origin_screen[0] + 10, origin_screen[1] + 10))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                # Handle iteration editing mode
                if self.editing_iterations_ui:
                    if event.key == pygame.K_RETURN:
                        # Apply the entered value
                        try:
                            new_iterations = int(self.iteration_input) if self.iteration_input else self.fractal_iterations
                            self.fractal_iterations = max(1, min(100000, new_iterations))
                            self.generate_fractal(force_full=True)
                        except ValueError:
                            pass  # Keep current value if invalid input
                        self.editing_iterations_ui = False
                        self.iteration_input = ""
                    elif event.key == pygame.K_ESCAPE:
                        # Cancel editing
                        self.editing_iterations_ui = False
                        self.iteration_input = ""
                    elif event.key == pygame.K_BACKSPACE:
                        # Delete last character
                        self.iteration_input = self.iteration_input[:-1]
                    elif event.unicode.isdigit() and len(self.iteration_input) < 6:  # Limit input length
                        # Add digit
                        self.iteration_input += event.unicode
                # Handle normal mode
                else:
                    if event.key == pygame.K_e:
                        # Toggle UI visibility
                        self.show_ui = not self.show_ui
                        # Deselect rectangles when hiding UI
                        if not self.show_ui:
                            for rect in self.rectangles:
                                rect.selected = False
                            self.selected_rect = None

                    elif event.key == pygame.K_SPACE and self.show_ui:
                        mouse_px, mouse_py = pygame.mouse.get_pos()
                        mx, my = screen_to_norm(mouse_px, mouse_py)
                        self.rectangles.append(Rectangle(0.3, 0.3, 0, mx, my))

                    elif event.key == pygame.K_r and self.show_ui:
                        self.generate_random_ifs()

                    elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                        if self.selected_rect and self.show_ui:
                            self.rectangles.remove(self.selected_rect)
                            self.selected_rect = None
                            self.generate_fractal()

                    elif event.key == pygame.K_i and self.show_ui:
                        self.editing_iterations_ui = True
                        self.iteration_input = str(self.fractal_iterations)
                    
                    elif event.key == pygame.K_c and self.show_ui:
                        # Cycle through colormaps
                        colormaps = ["binary", "hot", "viridis", "magma", "afmhot", "inferno", "bone", "grey"]
                        current_idx = colormaps.index(self.colormap_name) if self.colormap_name in colormaps else 0
                        self.colormap_name = colormaps[(current_idx + 1) % len(colormaps)]
                        self.update_fractal_surface()  # Re-render with new colormap

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and self.show_ui:
                    self.is_editing = True  # Start editing mode
                    mouse_px, mouse_py = event.pos

                    # Check for handle interactions first
                    if self.selected_rect:
                        interaction = self.selected_rect.get_interaction_type(mouse_px, mouse_py)
                        if interaction:
                            self.interaction_mode = interaction
                            self.last_mouse_pos = event.pos

                            if interaction == "rotate":
                                # Calculate initial angle offset for rotation
                                center_screen = norm_to_screen(self.selected_rect.center_x, self.selected_rect.center_y)
                                dx = mouse_px - center_screen[0]
                                dy = mouse_py - center_screen[1]
                                # Negate dy to account for flipped Y-axis in screen coordinates
                                mouse_angle = math.atan2(-dy, dx)
                                self.initial_angle = mouse_angle - self.selected_rect.theta

                            elif interaction.startswith("scale"):
                                self.initial_scale = (self.selected_rect.scale_x, self.selected_rect.scale_y)

                            return True

                    # Select rectangle
                    clicked_rect = None
                    for rect in reversed(self.rectangles):
                        if rect.contains_point(mouse_px, mouse_py):
                            clicked_rect = rect
                            break

                    # Update selection
                    for rect in self.rectangles:
                        rect.selected = False

                    if clicked_rect:
                        clicked_rect.selected = True
                        self.selected_rect = clicked_rect
                    else:
                        self.selected_rect = None
                    self.interaction_mode = None

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.is_editing = False  # End editing mode
                    self.interaction_mode = None
                    self.last_mouse_pos = None
                    # Generate full quality fractal after editing
                    if self.rectangles:
                        self.generate_fractal(force_full=True)

            elif event.type == pygame.MOUSEMOTION:
                if self.interaction_mode and self.selected_rect and self.last_mouse_pos and self.show_ui:
                    mouse_px, mouse_py = event.pos

                    if self.interaction_mode == "move":
                        # Calculate movement in normalized coordinates
                        dx_px = mouse_px - self.last_mouse_pos[0]
                        dy_px = mouse_py - self.last_mouse_pos[1]
                        
                        # Convert pixel delta to normalized delta
                        dx_norm = (dx_px / SCREEN_WIDTH) * NORM_WIDTH
                        dy_norm = -(dy_px / SCREEN_HEIGHT) * NORM_HEIGHT  # Negative because screen Y is flipped
                        
                        self.selected_rect.center_x += dx_norm
                        self.selected_rect.center_y += dy_norm
                        
                        # Update fractal in real-time
                        self.generate_fractal()

                    elif self.interaction_mode == "rotate":
                        # Calculate rotation based on mouse position relative to center
                        center_screen = norm_to_screen(self.selected_rect.center_x, self.selected_rect.center_y)
                        dx = mouse_px - center_screen[0]
                        dy = mouse_py - center_screen[1]
                        # Negate dy to account for flipped Y-axis in screen coordinates
                        current_mouse_angle = math.atan2(-dy, dx)
                        self.selected_rect.theta = current_mouse_angle - self.initial_angle
                        
                        # Update fractal in real-time
                        self.generate_fractal()

                    elif self.interaction_mode.startswith("scale_corner"):
                        try:
                            # Transform mouse position to local coordinates
                            M_rigid = self.selected_rect.rigid_motion_matrix
                            M_rigid_inv = np.linalg.inv(M_rigid)
                            mx_norm, my_norm = screen_to_norm(mouse_px, mouse_py)
                            mouse_world = np.array([mx_norm, my_norm, 1])
                            mouse_local = M_rigid_inv @ mouse_world

                            # Set scale based on mouse position in local coordinates
                            self.selected_rect.scale_x = mouse_local[0]
                            self.selected_rect.scale_y = mouse_local[1]
                            
                            # Update fractal in real-time
                            self.generate_fractal()
                        except np.linalg.LinAlgError:
                            pass

                    elif self.interaction_mode.startswith("scale_edge"):
                        try:
                            # Transform mouse position to local coordinates
                            M_rigid = self.selected_rect.rigid_motion_matrix
                            M_rigid_inv = np.linalg.inv(M_rigid)
                            mx_norm, my_norm = screen_to_norm(mouse_px, mouse_py)
                            mouse_world = np.array([mx_norm, my_norm, 1])
                            mouse_local = M_rigid_inv @ mouse_world

                            edge_index = int(self.interaction_mode.split("_")[-1])

                            # Scale along the appropriate axis based on edge
                            if edge_index == 0 or edge_index == 2:  # bottom or top edge - scale Y
                                self.selected_rect.scale_y = mouse_local[1]
                            else:  # left or right edge - scale X
                                self.selected_rect.scale_x = mouse_local[0]
                            
                            # Update fractal in real-time
                            self.generate_fractal()
                        except np.linalg.LinAlgError:
                            pass

                    self.last_mouse_pos = event.pos

        return True

    def run(self):
        running = True
        while running:
            running = self.handle_events()

            self.screen.fill(BLACK)

            # Continue fractal rendering when not editing
            if not self.is_editing:
                self.continue_fractal_rendering()

            # Draw fractal surface FIRST (background)
            self.screen.blit(self.fractal_surface, (0, 0))
            
            # Draw grid AFTER fractal surface (so it appears on top)
            if self.show_ui:
                self.draw_grid()
            
            # Draw rectangles on top of everything
            if self.show_ui:
                for rect in self.rectangles:
                    rect.draw(self.screen, show_ui=self.show_ui)

            # Show instructions and current status only if UI is visible
            if self.show_ui:
                font = pygame.font.Font(None, 24)
                instructions = [
                    "SPACE: Add rectangle at mouse",
                    "R: Generate random IFS rectangles", 
                    "DELETE: Remove selected rectangle",
                    f"Iterations: {self.fractal_iterations} (Press 'I' to edit)",
                    f"Colormap: {self.colormap_name} (Press 'C' to cycle)",
                    "Drag handles to scale/rotate (real-time fractal update)",
                    "E: Toggle UI visibility (hide/show all controls)"
                ]
                for i, instruction in enumerate(instructions):
                    text = font.render(instruction, True, WHITE)
                    self.screen.blit(text, (10, 10 + i * 25))

                if self.editing_iterations_ui:
                    input_text = font.render(f"Enter iterations: {self.iteration_input}_ (ENTER to apply, ESC to cancel)", True, YELLOW)
                    self.screen.blit(input_text, (10, 185))

                # Show editing status
                if self.is_editing:
                    status_text = font.render(f"EDITING (using {self.editing_iterations} iterations)", True, YELLOW)
                    self.screen.blit(status_text, (10, SCREEN_HEIGHT - 60))

                # Display coordinate system info
                font_small = pygame.font.Font(None, 18)
                coord_info = f"Coordinate system: X: [{-NORM_WIDTH/2:.1f}, {NORM_WIDTH/2:.1f}], Y: [{-NORM_HEIGHT/2:.1f}, {NORM_HEIGHT/2:.1f}]"
                coord_text = font_small.render(coord_info, True, LIGHT_GRAY)
                self.screen.blit(coord_text, (10, SCREEN_HEIGHT - 25))
            else:
                # Show minimal UI toggle hint when UI is hidden
                font_small = pygame.font.Font(None, 18)
                hint_text = font_small.render("Press 'E' to show UI", True, LIGHT_GRAY)
                self.screen.blit(hint_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    editor = RectangleEditor()
    editor.run()