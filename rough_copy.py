import os
from dataclasses import dataclass


@dataclass
class UtilsConfig:
    os.makedirs("src/results/models", exist_ok=True)
    os.makedirs("src/results/plots", exist_ok=True)

    model_dir = os.path.join("src/results/models")
    plot_dir = os.path.join("src/results/plots")


if __name__ == '__main__':
    # path =os.path.join(UtilsConfig().model_dir,f"model.pth")
    path:str = os.path.dirname(UtilsConfig().plot_dir)
    print(path)