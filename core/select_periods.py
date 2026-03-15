import pandas as pd

def latest_completed_week(week_df):
    return week_df["date"].max()