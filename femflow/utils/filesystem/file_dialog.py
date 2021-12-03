import tkinter as tk
from tkinter import filedialog
from typing import Any

import igl
from loguru import logger
from viz.mesh import Mesh


def file_dialog_open() -> str:
    logger.debug("Opening file dialog")
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    if not file_path:
        logger.info("Aborted opening file")
        return

    logger.info(f"Loading mesh: {file_path}")
    return str(file_path)


def file_dialog_save(content: Any):
    logger.debug("Opening file dialog")
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfile(mode="w")

    if not file_path:
        logger.warning("Aborted saving file")
        return

    logger.info(f"Saving mesh: {file_path.name}")

    try:
        file_path.write(str(content))
        file_path.close()
    except Exception as e:
        logger.error(f"Error writing file: {e}")


def file_dialog_save_mesh(content: Mesh):
    logger.debug("Opening file dialog")
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename()

    if not file_path:
        logger.warning("Aborted saving file")
        return

    logger.info(f"Saving mesh: {file_path}")

    v, f = content.unroll_to_igl_mesh()

    try:
        igl.write_triangle_mesh(file_path, v, f)
    except Exception as e:
        logger.error(f"Write failed with error: {e}")
