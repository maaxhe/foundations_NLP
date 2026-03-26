"""
character_sampler.py
Loads the D&D character dataset and samples realistic NPCs
based on learned distributions (race, class, background, stats).
"""

import os
import random
import numpy as np
import pandas as pd

# D&D name parts for random name generation
FIRST_NAMES = [
    "Aldric", "Mira", "Theron", "Seraphina", "Gruff", "Lyra", "Borin", "Elara",
    "Torvin", "Nessa", "Drakon", "Sylvara", "Garett", "Isolde", "Bryn", "Cassian",
    "Hilda", "Fenwick", "Zara", "Oswin", "Petra", "Colt", "Wren", "Dagmar",
    "Rook", "Elowen", "Vance", "Sable", "Jasper", "Imara",
]

LAST_NAMES = [
    "Stonebrew", "Ashveil", "Ironforge", "Brightwater", "Coldmere", "Thornwood",
    "Duskmantle", "Silverbell", "Grimshaw", "Oakenshield", "Blackthorn", "Meadowlark",
    "Ravenscar", "Goldenleaf", "Embercroft", "Nighthollow", "Stormridge", "Fairweather",
    "Deepwater", "Cinderheart",
]

ALIGNMENT_MAP = {
    "LG": "Lawful Good", "NG": "Neutral Good", "CG": "Chaotic Good",
    "LN": "Lawful Neutral", "N": "True Neutral", "CN": "Chaotic Neutral",
    "LE": "Lawful Evil", "NE": "Neutral Evil", "CE": "Chaotic Evil",
}

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "data_sets", "dnd_character_database", "dnd_chars_all.csv"
)


class CharacterSampler:
    """Learns distributions from the D&D dataset and samples new characters."""

    def __init__(self, csv_path: str = DATA_PATH):
        self.df = self._load(csv_path)
        self._build_distributions()

    def _load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep=";", low_memory=False)
        # Keep only numeric stat columns
        stat_cols = ["Str", "Dex", "Con", "Int", "Wis", "Cha", "HP", "AC", "level"]
        for col in stat_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Filter out rows with missing core fields
        df = df.dropna(subset=["race", "justClass"])
        # Normalize race: use processedRace when available
        df["race_clean"] = df["processedRace"].fillna(df["race"])
        # Primary class (first class listed for multiclass chars)
        df["primary_class"] = df["justClass"].str.split("|").str[0].str.strip()
        return df

    def _build_distributions(self):
        """Pre-compute weighted distributions for sampling."""
        self.race_dist = self.df["race_clean"].value_counts(normalize=True)
        self.class_dist = self.df["primary_class"].value_counts(normalize=True)
        self.background_dist = self.df["background"].dropna().value_counts(normalize=True)

        # Alignment: filter to known codes
        known = list(ALIGNMENT_MAP.keys())
        align_counts = self.df["processedAlignment"].dropna()
        align_counts = align_counts[align_counts.isin(known)].value_counts(normalize=True)
        self.alignment_dist = align_counts if len(align_counts) > 0 else pd.Series({"NG": 1.0})

        # Per-class stat means and stds
        stat_cols = ["Str", "Dex", "Con", "Int", "Wis", "Cha"]
        self.stat_params = {}
        for cls, grp in self.df.groupby("primary_class"):
            self.stat_params[cls] = {
                col: (grp[col].mean(), grp[col].std()) for col in stat_cols
            }
        # Fallback global stats
        self.global_stat_params = {
            col: (self.df[col].mean(), self.df[col].std()) for col in stat_cols
        }
        # HP and AC global
        self.hp_params = (self.df["HP"].dropna().mean(), self.df["HP"].dropna().std())
        self.ac_params = (self.df["AC"].dropna().mean(), self.df["AC"].dropna().std())
        self.level_params = (self.df["level"].dropna().mean(), self.df["level"].dropna().std())

    def _weighted_sample(self, dist: pd.Series) -> str:
        return np.random.choice(dist.index, p=dist.values)

    def _sample_stat(self, mean: float, std: float, lo: int = 1, hi: int = 20) -> int:
        if np.isnan(mean):
            mean = 10.0
        if np.isnan(std):
            std = 2.0
        val = int(round(np.random.normal(mean, max(std, 1.0))))
        return max(lo, min(hi, val))

    def sample_character(self) -> dict:
        """Sample a single NPC character dict from learned distributions."""
        race = self._weighted_sample(self.race_dist)
        primary_class = self._weighted_sample(self.class_dist)
        background = self._weighted_sample(self.background_dist)
        alignment_code = self._weighted_sample(self.alignment_dist)
        alignment = ALIGNMENT_MAP.get(alignment_code, "True Neutral")

        # Stats: use per-class params if available
        params = self.stat_params.get(primary_class, self.global_stat_params)
        stats = {col: self._sample_stat(*params[col]) for col in ["Str", "Dex", "Con", "Int", "Wis", "Cha"]}

        level = max(1, min(20, int(round(np.random.normal(*self.level_params)))))
        hp = max(1, int(round(np.random.normal(*self.hp_params))))
        ac = max(5, min(25, int(round(np.random.normal(*self.ac_params)))))

        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

        return {
            "name": name,
            "race": race,
            "primary_class": primary_class,
            "background": background,
            "alignment": alignment,
            "alignment_code": alignment_code,
            "level": level,
            "HP": hp,
            "AC": ac,
            **stats,
        }


if __name__ == "__main__":
    sampler = CharacterSampler()
    char = sampler.sample_character()
    for k, v in char.items():
        print(f"{k:20s}: {v}")
