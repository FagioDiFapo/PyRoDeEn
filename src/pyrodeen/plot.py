import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig_1d = None
lines_1d = None
frames_1d = []

def animate_1d(frame_idx):
    frame = frames_1d[frame_idx]
    lines_1d[0].set_data(frame['x'], frame['r'])
    lines_1d[1].set_data(frame['x'], frame['u'])
    lines_1d[2].set_data(frame['x'], frame['p'])
    lines_1d[3].set_data(frame['x'], frame['E'])
    return lines_1d

def save_1d_animation(filename="euler_1d.mp4", fps=10):
    global lines_1d, fig_1d, frames_1d
    anim = animation.FuncAnimation(fig_1d, animate_1d, frames=len(frames_1d), interval=50, blit=True)
    anim.save(filename, writer='ffmpeg', fps=20)
    print(f"Video saved to {filename}")

    # Restore the final frame data to the plot after animation completes
    final_frame = frames_1d[-1]
    lines_1d[0].set_data(final_frame['x'], final_frame['r'])
    lines_1d[1].set_data(final_frame['x'], final_frame['u'])
    lines_1d[2].set_data(final_frame['x'], final_frame['p'])
    lines_1d[3].set_data(final_frame['x'], final_frame['E'])
    plt.draw()
    plt.pause(0.1)

def plot_euler_1d(x, r, u, p, E, record = False):
    global lines_1d, fig_1d
    if lines_1d is None:
        plt.ion()
        fig_1d, axs = plt.subplots(2, 2, num=2)
        lines_1d = [
            axs[0, 0].plot([], [], ".b")[0],
            axs[0, 1].plot([], [], ".m")[0],
            axs[1, 0].plot([], [], ".k")[0],
            axs[1, 1].plot([], [], ".r")[0],
        ]
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
        fig_1d.suptitle("SSP-RK2 TVD-MUSCL Euler Eqns. (2D Sod Tube)", fontsize=10)
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
        plt.tight_layout()

    lines_1d[0].set_data(x, r)
    lines_1d[1].set_data(x, u)
    lines_1d[2].set_data(x, p)
    lines_1d[3].set_data(x, E)

    if record:
            frames_1d.append({
                'x': x.copy(),
                'r': r.copy(),
                'u': u.copy(),
                'p': p.copy(),
                'E': E.copy()
            })

    # draw
    plt.draw()
    plt.pause(0.1)

    # Return whether the figure still exists (False if closed)
    return fig_1d, plt.fignum_exists(fig_1d.number), lines_1d