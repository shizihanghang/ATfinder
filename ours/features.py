from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AAINDEX_CSV = PROJECT_ROOT / "resources" / "aaindex.csv"
MAX_LEN = 50
PHYS_IN_DIM = 105

AA_dict = {'A': 0,
 'R': 1,
 'N': 2,
 'D': 3,
 'C': 4,
 'Q': 5,
 'E': 6,
 'G': 7,
 'H': 8,
 'I': 9,
 'L': 10,
 'K': 11,
 'M': 12,
 'F': 13,
 'P': 14,
 'S': 15,
 'T': 16,
 'W': 17,
 'Y': 18,
 'V': 19}

AA_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

parameter_list = ['ARGP820101',
 'EISD860102',
 'FAUJ830101',
 'FAUJ880108',
 'FAUJ880109',
 'CHAM830108',
 'FAUJ880111',
 'MITS020101',
 'GOLD730101',
 'GRAR740101',
 'GRAR740103',
 'JOND750101',
 'PONP930101',
 'LAWE840101',
 'NOZY710101',
 'ROBB790101',
 'ROSM880101',
 'ROSM880103',
 'FASG760101',
 'SIMZ760101',
 'TAKK010101',
 'VELV850101',
 'WILM950102',
 'WILM950104',
 'ZIMJ680103',
 'CHAM830106',
 'CHAM830107',
 'LEVM760107',
 'RICJ880106',
 'RICJ880107',
 'RICJ880108',
 'CHAM830105',
 'COSI940101',
 'GARJ730101',
 'CHOP780204',
 'OOBM850102',
 'PALJ810113',
 'RADA880102',
 'BASU050101',
 'FAUJ880110',
 'FAUJ880112',
 'KLEP840101',
 'CHAM820102',
 'KRIW790103',
 'EISD860103']

hydrophobicity = {'A': [100, 13, 60, 33, 98, 64, 39, 96, 71, 91, 93, 35, 89, 87, 89, 94, 98, 98, 94, 94],
 'R': [13, 100, 53, 81, 11, 49, 74, 17, 42, 4, 6, 78, 2, 0, 24, 19, 16, 11, 19, 7],
 'N': [60, 53, 100, 73, 58, 96, 79, 64, 89, 51, 52, 75, 49, 47, 71, 66, 63, 58, 66, 54],
 'D': [33, 81, 73, 100, 30, 68, 94, 36, 61, 23, 25, 98, 21, 19, 44, 39, 35, 31, 38, 26],
 'C': [98, 11, 58, 30, 100, 62, 36, 94, 69, 93, 95, 33, 91, 89, 86, 91, 95, 99, 92, 96],
 'Q': [64, 49, 96, 68, 62, 100, 74, 68, 93, 55, 57, 71, 53, 51, 76, 71, 67, 63, 70, 58],
 'E': [39, 74, 79, 94, 36, 74, 100, 43, 68, 29, 31, 96, 27, 26, 50, 45, 41, 37, 44, 33],
 'G': [96, 17, 64, 36, 94, 68, 43, 100, 75, 87, 89, 39, 85, 83, 93, 98, 99, 94, 98, 90],
 'H': [71, 42, 89, 61, 69, 93, 68, 75, 100, 62, 64, 64, 60, 58, 83, 78, 74, 69, 77, 65],
 'I': [91, 4, 51, 23, 93, 55, 29, 87, 62, 100, 98, 26, 98, 96, 79, 84, 88, 93, 85, 97],
 'L': [93, 6, 52, 25, 95, 57, 31, 89, 64, 98, 100, 27, 96, 94, 81, 86, 90, 94, 87, 99],
 'K': [35, 78, 75, 98, 33, 71, 96, 39, 64, 26, 27, 100, 24, 22, 46, 41, 38, 33, 41, 29],
 'M': [89, 2, 49, 21, 91, 53, 27, 85, 60, 98, 96, 24, 100, 98, 78, 83, 86, 91, 83, 95],
 'F': [87, 0, 47, 19, 89, 51, 26, 83, 58, 96, 94, 22, 98, 100, 76, 81, 84, 89, 81, 93],
 'P': [89, 24, 71, 44, 86, 76, 50, 93, 83, 79, 81, 46, 78, 76, 100, 95, 91, 87, 94, 83],
 'S': [94, 19, 66, 39, 91, 71, 45, 98, 78, 84, 86, 41, 83, 81, 95, 100, 96, 92, 99, 88],
 'T': [98, 16, 63, 35, 95, 67, 41, 99, 74, 88, 90, 38, 86, 84, 91, 96, 100, 96, 97, 91],
 'W': [98, 11, 58, 31, 99, 63, 37, 94, 69, 93, 94, 33, 91, 89, 87, 92, 96, 100, 93, 96],
 'Y': [94, 19, 66, 38, 92, 70, 44, 98, 77, 85, 87, 41, 83, 81, 94, 99, 97, 93, 100, 88],
 'V': [94, 7, 54, 26, 96, 58, 33, 90, 65, 97, 99, 29, 95, 93, 83, 88, 91, 96, 88, 100]}

blosum62 = {'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
 'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
 'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
 'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
 'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
 'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
 'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
 'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
 'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
 'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
 'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
 'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
 'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
 'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
 'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
 'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
 'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
 'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
 'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
 'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4]}

PAM120 = {'A': [3, -3, 0, 0, -3, -1, 0, 1, -3, -1, -3, -2, -2, -4, 1, 1, 1, -7, -4, 0],
 'R': [-3, 6, -1, -3, -4, 1, -3, -4, 1, -2, -4, 2, -1, -4, -1, -1, -2, 1, -6, -3],
 'N': [0, -1, 4, 2, -5, 0, 1, 0, 2, -2, -4, 1, -3, -4, -2, 1, 0, -5, -2, -3],
 'D': [0, -3, 2, 5, -7, 1, 3, 0, 0, -3, -5, -1, -4, -7, -2, 0, -1, -8, -5, -3],
 'C': [-3, -4, -5, -7, 9, -7, -7, -5, -4, -3, -7, -7, -6, -6, -3, -1, -3, -8, -1, -2],
 'Q': [-1, 1, 0, 1, -7, 6, 2, -3, 3, -3, -2, 0, -1, -6, 0, -2, -2, -6, -5, -3],
 'E': [0, -3, 1, 3, -7, 2, 5, -1, -1, -3, -4, -1, -4, -6, -1, -1, -2, -8, -4, -3],
 'G': [1, -4, 0, 0, -5, -3, -1, 5, -4, -4, -5, -3, -4, -5, -2, 1, -1, -8, -6, -2],
 'H': [-3, 1, 2, 0, -4, 3, -1, -4, 7, -4, -3, -2, -4, -2, -1, -2, -3, -5, -1, -3],
 'I': [-1, -2, -2, -3, -3, -3, -3, -4, -4, 6, 1, -2, 1, 0, -3, -2, 0, -7, -2, 3],
 'L': [-3, -4, -4, -5, -7, -2, -4, -5, -3, 1, 5, -4, 3, 0, -3, -4, -3, -5, -3, 1],
 'K': [-2, 2, 1, -1, -7, 0, -1, -3, -2, -2, -4, 5, 0, -6, -2, -1, -1, -5, -6, -4],
 'M': [-2, -1, -3, -4, -6, -1, -4, -4, -4, 1, 3, 0, 8, -1, -3, -2, -1, -7, -4, 1],
 'F': [-4, -4, -4, -7, -6, -6, -6, -5, -2, 0, 0, -6, -1, 8, -5, -3, -4, -1, 4, -3],
 'P': [1, -1, -2, -2, -3, 0, -1, -2, -1, -3, -3, -2, -3, -5, 6, 1, -1, -7, -6, -2],
 'S': [1, -1, 1, 0, -1, -2, -1, 1, -2, -2, -4, -1, -2, -3, 1, 3, 2, -2, -3, -2],
 'T': [1, -2, 0, -1, -3, -2, -2, -1, -3, 0, -3, -1, -1, -4, -1, 2, 4, -6, -3, 0],
 'W': [-7, 1, -5, -8, -8, -6, -8, -8, -5, -7, -5, -5, -7, -1, -7, -2, -6, 12, -1, -8],
 'Y': [-4, -6, -2, -5, -1, -5, -4, -6, -1, -2, -3, -6, -4, 4, -6, -3, -3, -1, 8, -3],
 'V': [0, -3, -3, -3, -2, -3, -3, -2, -3, 3, 1, -4, 1, -3, -2, -2, 0, -8, -3, 5]}

def build_phys_feature_table(
    device: torch.device, aaindex_path: Path | None = None
) -> torch.Tensor:
    aaindex_path = AAINDEX_CSV if aaindex_path is None else Path(aaindex_path)
    if not aaindex_path.exists():
        raise FileNotFoundError(
            f"AAindex file not found: {aaindex_path}"
        )
    frame = pd.read_csv(aaindex_path, index_col=0)
    missing = [name for name in parameter_list if name not in frame.index]
    if missing:
        raise ValueError(f"AAindex file is missing required descriptors: {missing}")
    aaindex_rows = []
    for name in parameter_list:
        values = frame.loc[name].to_numpy(dtype="float32")
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if values.shape[0] != 20:
            raise ValueError(f"AAindex descriptor {name} has {values.shape[0]} values; expected 20")
        aaindex_rows.append(torch.as_tensor(values, device=device))
    aaindex = torch.stack(aaindex_rows, dim=0)

    rows = []
    for amino_acid in AA_type:
        index = AA_dict[amino_acid]
        row = torch.cat([
            aaindex[:, index].float(),
            torch.tensor(hydrophobicity[amino_acid], dtype=torch.float32, device=device),
            torch.tensor(blosum62[amino_acid], dtype=torch.float32, device=device),
            torch.tensor(PAM120[amino_acid], dtype=torch.float32, device=device),
        ])
        rows.append(row)
    table = torch.stack(rows, dim=0)
    if table.shape != (20, PHYS_IN_DIM):
        raise RuntimeError(f"Unexpected physicochemical table shape: {tuple(table.shape)}")
    return table


def encode_sequences_phys(sequences: list[str], phys_table: torch.Tensor,
                          max_len: int = MAX_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(sequences)
    padding = phys_table.shape[0]
    indices = np.full((batch, max_len), padding, dtype=np.int64)
    lengths = []
    for row, sequence in enumerate(sequences):
        clean = str(sequence).strip().upper()
        length = min(len(clean), max_len)
        lengths.append(length)
        for position, amino_acid in enumerate(clean[:max_len]):
            indices[row, position] = AA_dict.get(amino_acid, padding)
    lookup = torch.cat([phys_table, phys_table.new_zeros((1, phys_table.shape[1]))])
    indices = torch.as_tensor(indices, device=phys_table.device)
    output = lookup.index_select(0, indices.reshape(-1)).reshape(batch, max_len, phys_table.shape[1])
    return output, torch.tensor(lengths, dtype=torch.long, device=phys_table.device)
