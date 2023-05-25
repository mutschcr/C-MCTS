import os
import pickle
import matplotlib.pyplot as plt
from celluloid import Camera


def render(render_path):
    num_rocks = 8
    pickle_files = [f for f in os.listdir(render_path) if f.endswith('.pickle')]
    for f in pickle_files:
        # Load data (deserialize)
        with open(render_path + f, 'rb') as handle:
            episode_no = f.split("_")[1].split(".")[0]
            try:
                l_positions = pickle.load(handle)
            except:
                print("failed to load episode " + str(episode_no))
                continue

        fig, ax = plt.subplots()
        camera = Camera(fig)
        R = 0
        C = 0
        discount = 1.0
        for t, pos in enumerate(l_positions):
            agent = plt.Circle((pos['agent_x'] + 0.5, pos['agent_y'] + 0.5), 0.15, color='blue')
            action = pos['action']
            r = pos['reward']
            c = pos['cost']
            if action == 4:
                ax.text(pos['agent_x'] + 0.5, pos['agent_y'] + 0.5, "S", fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
            elif action > 4:  # rocks are being measured
                x_initial = pos['agent_x'] + 0.5
                y_initial = pos['agent_y'] + 0.5
                rock_index = action - 4
                x_final = pos[f'rock{rock_index}_x'] + 0.5
                y_final = pos[f'rock{rock_index}_y'] + 0.5
                dx = x_final - x_initial  # length of arrow in x direction
                dy = y_final - y_initial  # length of arrow in y direction
                arrow = plt.arrow(x_initial, y_initial, dx, dy)
                ax.add_patch(arrow)
            ax.add_patch(agent)
            for i in range(num_rocks):
                rock = [value for key, value in pos.items() if f"rock{i + 1}" in key]
                if not rock[3]:  # rock is not yet collected
                    if rock[2]:  # rock is valuable
                        ax.add_patch(plt.Circle((rock[0] + 0.5, rock[1] + 0.5), 0.05, color="green"))
                    else:
                        ax.add_patch(plt.Circle((rock[0] + 0.5, rock[1] + 0.5), 0.05, color="red"))
            ax.set_xlim([0, 7])
            ax.set_ylim([0, 7])
            R += r * discount
            C += c * discount
            discount *= 0.95
            ax.text(2, 7.2, f"t={t},r={r},R={round(R, 2)},C={round(C, 2)}")
            ax.grid(True)
            ax.set_aspect('equal')
            camera.snap()

        anim = camera.animate(blit=False, interval=700)
        anim.save(render_path + f"episode_{episode_no}.gif")
