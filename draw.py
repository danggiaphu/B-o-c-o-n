"""Utilities for drawing chemical structures from SMILES strings.

This module uses RDKit to convert SMILES into 2D structure drawings.

Example:
    python draw.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --output aspirin.png
    python draw.py --smiles "C1=CC=CC=C1" --svg --output benzene.svg

Requirements:
    rdkit
    pillow
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError as exc:
    raise ImportError(
        "RDKit is required to use draw.py. Install it with `pip install rdkit-pypi` "
        "or via conda if you are in a conda environment."
    ) from exc

try:
    from PIL import Image
except ImportError as exc:
    raise ImportError("Pillow is required to export images. Install it with `pip install pillow`.") from exc


def get_mol_from_smiles(smiles: str) -> Chem.Mol:
    smiles = smiles.strip()
    if not smiles:
        raise ValueError("SMILES string must not be empty.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        # Kekulization may fail on some aromatic molecules; continue with the original molecule.
        pass

    return mol


def draw_molecule_image(smiles: str, size: Tuple[int, int] = (320, 320)) -> Image.Image:
    mol = get_mol_from_smiles(smiles)
    image = Draw.MolToImage(mol, size=size)
    if not isinstance(image, Image.Image):
        raise RuntimeError("RDKit did not return a PIL.Image object for the molecule drawing.")
    return image


def draw_molecule_svg(smiles: str, size: Tuple[int, int] = (320, 320)) -> str:
    mol = get_mol_from_smiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg


def save_molecule_image(smiles: str, output_path: Path, size: Tuple[int, int] = (320, 320)) -> None:
    output_path = output_path.expanduser()
    if output_path.suffix.lower() == ".svg":
        svg = draw_molecule_svg(smiles, size=size)
        output_path.write_text(svg, encoding="utf-8")
    else:
        image = draw_molecule_image(smiles, size=size)
        image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a molecular structure from a SMILES string using RDKit.")
    parser.add_argument("--smiles", required=True, help="SMILES string of the molecule.")
    parser.add_argument("--output", type=Path, help="Output file path. If omitted, the drawing is written to stdout for SVG or saved as molecule.png.")
    parser.add_argument("--svg", action="store_true", help="Export the structure as SVG instead of PNG.")
    parser.add_argument("--width", type=int, default=320, help="Image width in pixels.")
    parser.add_argument("--height", type=int, default=320, help="Image height in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size = (args.width, args.height)

    if args.output:
        output_path = args.output
        if args.svg and output_path.suffix.lower() not in {".svg", ""}:
            raise ValueError("When using --svg, output path should have a .svg extension.")
        if not args.svg and output_path.suffix == "":
            output_path = output_path.with_suffix(".png")
        save_molecule_image(args.smiles, output_path, size=size)
        print(f"Saved molecule drawing to {output_path}")
    else:
        if args.svg:
            svg = draw_molecule_svg(args.smiles, size=size)
            print(svg)
        else:
            output_path = Path("molecule.png")
            save_molecule_image(args.smiles, output_path, size=size)
            print(f"Saved molecule drawing to {output_path}")


if __name__ == "__main__":
    main()
