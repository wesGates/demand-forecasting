"""Smoke check: build the study panel and print what came back."""

from src.step1_problem import Config
from src.step2_data import load_panel

STUDY_ITEMS = ("FOODS_3_120", "FOODS_3_681", "FOODS_3_282")

if __name__ == "__main__":
    cfg = Config(item_ids=STUDY_ITEMS)
    print(cfg.describe(), "\n")
    load_panel(cfg)
