import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyrodeen.solutions import EulerSolution1D

fig_1d = None
lines_1d = None
frames_1d = []


def create_1d_plot(reference: EulerSolution1D = None):
    global fig_1d, lines_1d
    fig, axs = plt.subplots(2, 2, dpi=100)
    lines = [
        axs[0, 0].plot([], [], ".b")[0],
        axs[0, 1].plot([], [], ".m")[0],
        axs[1, 0].plot([], [], ".k")[0],
        axs[1, 1].plot([], [], ".r")[0],
    ]
    if reference is not None:
        x, r, u, p, E = reference.x, reference.rho, reference.u, reference.p, reference.E
        axs[0, 0].plot(x, r, "-k", label="Exact")
        axs[0, 1].plot(x, u, "-k", label="Exact")
        axs[1, 0].plot(x, p, "-k", label="Exact")
        axs[1, 1].plot(x, E, "-k", label="Exact")
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel(r"$\rho$")
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("u")
    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_ylabel("p")
    axs[1, 1].set_xlabel("x")
    axs[1, 1].set_ylabel("E")
    # fig.suptitle("SSP-RK2 TVD-MUSCL Euler Eqns. (2D Sod Tube)", fontsize=10)
    axs[0, 0].set_xlim([0, 1])
    axs[0, 0].set_ylim([0, 1.1])
    axs[0, 1].set_xlim([0, 1])
    axs[0, 1].set_ylim([-0.1, 1.1])
    axs[1, 0].set_xlim([0, 1])
    axs[1, 0].set_ylim([0, 1.1])
    axs[1, 1].set_xlim([0, 1])
    axs[1, 1].set_ylim([1.5, 3.5])
    for axrow in axs:
        for ax in axrow:
            ax.legend()

    fig_1d = fig
    lines_1d = lines

    return fig, axs, lines


def animate_1d(frame_idx, lines):
    frame = frames_1d[frame_idx]
    lines[0].set_data(frame["x"], frame["r"])
    lines[1].set_data(frame["x"], frame["u"])
    lines[2].set_data(frame["x"], frame["p"])
    lines[3].set_data(frame["x"], frame["E"])
    return lines


def save_1d_animation(reference, filename="euler_1d.mp4", fps=10):
    global frames_1d
    fig, _, lines = create_1d_plot(reference)

    anim = animation.FuncAnimation(
        fig, animate_1d, frames=len(frames_1d), fargs=(lines,), interval=50, blit=True
    )
    anim.save(filename, writer="ffmpeg", fps=20)
    print(f"Video saved to {filename}")


def plot_euler_1d(solution: EulerSolution1D):
    global lines_1d, fig_1d
    if lines_1d is None:
        create_1d_plot(solution)
    else:
        x, r, u, p, E = solution.x, solution.rho, solution.u, solution.p, solution.E
        lines_1d[0].set_data(x, r)
        lines_1d[1].set_data(x, u)
        lines_1d[2].set_data(x, p)
        lines_1d[3].set_data(x, E)

    # draw
    plt.draw()
    plt.pause(0.03)

    # Return whether the figure still exists (False if closed)
    return plt.fignum_exists(fig_1d.number)


def record_euler_1d_frame(solution):
    global frames_1d
    x, r, u, p, E = solution.x, solution.rho, solution.u, solution.p, solution.E
    frames_1d.append(
        {"x": x.copy(), "r": r.copy(), "u": u.copy(), "p": p.copy(), "E": E.copy()}
    )


def prepare_plots(plot_1d=False, video_file_1d=None):
    pass
