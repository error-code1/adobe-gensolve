import numpy as np
import svgwrite
from svgpathtools import svg2paths2, Path, CubicBezier, Line, Arc
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def bezier_curve(t, P0, P1, P2, P3):
    return (1 - t)*3 * P0 + 3 * (1 - t)+2 * t * P1 + 3 * (1 - t) * t2 * P2 + t*3 * P3

def bezier_residuals(params, points):
    P0, P3 = points[0], points[-1]
    P1 = np.array(params[:2])
    P2 = np.array(params[2:])

    residuals = []
    n = len(points)

    for i, t in enumerate(np.linspace(0, 1, n)):
        B_t = bezier_curve(t, P0, P1, P2, P3)
        residuals.append(np.linalg.norm(B_t - points[i]))

    return residuals

def fit_bezier_curve(points, epochs=5):
    points = np.array([(p.real, p.imag) for p in points])
    P0, P3 = points[0], points[-1]

    P1_init = P0 + (P3 - P0) / 3
    P2_init = P0 + 2 * (P3 - P0) / 3

    initial_guess = np.concatenate((P1_init, P2_init))

    for epoch in range(epochs):
        result = minimize(lambda params: sum(bezier_residuals(params, points)),
                          initial_guess, method='BFGS')
        P1, P2 = np.split(result.x, 2)
        initial_guess = np.concatenate((P1, P2))  

    P0, P3 = points[0], points[-1]  
    return P0, P1, P2, P3

def arc_to_bezier(arc):
    """Convert an SVG arc to a cubic Bezier curve approximation."""
    start = arc.start
    end = arc.end
    radius = arc.radius
    large_arc_flag = arc.large_arc_flag
    sweep_flag = arc.sweep_flag
    
    return [CubicBezier(start, start + (end - start) / 3, end - (end - start) / 3, end)]

def regularize_svg_path(path, epochs=5):
    bezier_curves = []

    for segment in path:
        if isinstance(segment, CubicBezier):
            bezier_curves.append([segment.start, segment.control1, segment.control2, segment.end])
        elif isinstance(segment, Line):
            points = np.array([segment.start, segment.end])
            try:
                bezier_curve = fit_bezier_curve(points, epochs)
                bezier_curves.append(bezier_curve)
            except Exception as e:
                print(f"Error fitting Bezier curve: {e}")
        elif isinstance(segment, Arc):
            bezier_curves.extend(arc_to_bezier(segment))
        else:
            print(f"Unsupported segment type: {type(segment)}")

    return bezier_curves

def save_svg_from_curves(bezier_curves, svg_path, stroke_width=4):
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges', viewBox='0 0 1000 1000')
    for curve in bezier_curves:
        path_data = []
        path_data.append(f"M {curve[0].real:.4f},{curve[0].imag:.4f}")
        path_data.append(f"C {curve[1].real:.4f},{curve[1].imag:.4f} {curve[2].real:.4f},{curve[2].imag:.4f} {curve[3].real:.4f},{curve[3].imag:.4f}")
        dwg.add(dwg.path(d=' '.join(path_data), fill='none', stroke='black', stroke_width=stroke_width))
    dwg.save()

def plot_and_save_regularized_paths(svg_input_path, output_image_path, output_svg_path, epochs=5):
    paths, _, _ = svg2paths2(svg_input_path)
    regularized_paths = []

    for path in paths:
        try:
            regularized_curves = regularize_svg_path(path, epochs)
            print(f"Regularized curves for path: {regularized_curves}")  

            if regularized_curves:
                regularized_paths.extend(regularized_curves)

        except Exception as e:
            print(f"Error processing path: {e}")

    if regularized_paths:
        save_svg_from_curves(regularized_paths, output_svg_path, stroke_width=4)

    if regularized_paths:  
        plt.figure(figsize=(10, 10))
        for curve in regularized_paths:
            points = np.array([curve[0], curve[1], curve[2], curve[3]])
            t_values = np.linspace(0, 1, 100)
            curve_points = np.array([bezier_curve(t, *points) for t in t_values])
            plt.plot(curve_points.real, curve_points.imag, 'b-', linewidth=2, label='Regularized Bezier Curve')

        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig(output_image_path, dpi=600, bbox_inches='tight') 
        plt.show()
    else:
        print("No valid paths to plot.")

# Example usage
if __name__ == "__main__":
    svg_input_path = '/mnt/data/isolated.svg'  
    output_image_path = '/mnt/data/regularized_output.png'  
    output_svg_path = '/mnt/data/regularized_output.svg'  
    plot_and_save_regularized_paths(svg_input_path, output_image_path, output_svg_path, epochs=50)