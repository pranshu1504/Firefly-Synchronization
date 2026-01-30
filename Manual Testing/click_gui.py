import matplotlib.pyplot as plt
import json
import os

IMG_DIR = "validation_frames"
SAVE_DIR = "validation_points"
os.makedirs(SAVE_DIR, exist_ok=True)

def annotate_image(img_path):
    img = plt.imread(img_path)
    points = []

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Click on fireflies. Close window when done.")

    def onclick(event):
        if event.xdata and event.ydata:
            points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'rx')
            fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return points

# Run annotation
import glob
frames = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))

for f in frames:
    points = annotate_image(f)
    base = os.path.basename(f).replace(".png", "")
    with open(os.path.join(SAVE_DIR, base + ".json"), "w") as out:
        json.dump(points, out)

print("Done!")