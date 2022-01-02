import tkinter as tk
from tkinter import filedialog
from typing import Any

from femflow.viz.mesh import Mesh


def file_dialog_open() -> str:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    if not file_path:
        return ""

    return str(file_path)


def file_dialog_save(content: Any) -> bool:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfile(mode="w")

    if not file_path:
        return False

    try:
        file_path.write(str(content))
        file_path.close()
        return True
    except Exception as e:
        raise RuntimeError(f"Error writing file: {e}") from e


def file_dialog_save_mesh(content: Mesh) -> bool:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename()

    if not file_path:
        return False

    try:
        return content.save(file_path)
    except Exception as e:
        raise RuntimeError(f"file at {file_path} failed to save with error: {e}") from e
