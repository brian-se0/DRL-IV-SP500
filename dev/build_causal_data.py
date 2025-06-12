# build_causal_data.py
#
# Purpose:
# 1. Loads the manually collected SEC Section 31 fee rates.
# 2. Loads the main daily SPX ATM IV panel.
# 3. Merges the fee rate information into the IV panel, assigning the
#    correct prevailing fee rate to each day.
# 4. Identifies days on which fee rates changed and calculates the
#    magnitude of these changes.
# 5. Saves the augmented panel to a new CSV file for causal analysis.
#
# Assumes:
# - This script is located in the main project directory (e.g., 'ECON499/').
# - 'manual_sec_fees.csv' is present in 'ECON499/data_processed/'
#   and has columns 'EffectiveDate' and 'RatePerMillion'.
# - 'spx_daily_atm_iv.csv' is present in 'ECON499/data_processed/'.

import pandas as pd
from pathlib import Path

# --- Configuration ---
# Define project root. Assumes this script is in the project's root directory (e.g., 'ECON499/').
try:
    # If __file__ is defined, ROOT is the directory containing this script.
    # This is correct if build_causal_data.py is directly in ECON499/
    ROOT = Path(__file__).resolve().parent
except NameError:
    # Fallback for interactive environments (e.g., Jupyter notebook)
    # where __file__ might not be defined. Assumes the notebook or
    # interactive session is run from the project root (ECON499/).
    ROOT = Path(".").resolve()

# Input files
MANUAL_FEES_FILENAME = "manual_sec_fees.csv"  # Your manually curated SEC fees
IV_PANEL_FILENAME = "spx_daily_atm_iv.csv"  # Your main data panel

# Output file
OUTPUT_PANEL_FILENAME = "spx_daily_atm_iv_with_fees.csv"

# Data directories
DATA_PROCESSED_DIR = ROOT / "data_processed"


def prepare_causal_data():
    """
    Loads the daily IV panel, merges SEC fee rate data, identifies changes,
    and saves the augmented panel.
    """
    print(f"Project ROOT directory determined as: {ROOT}")
    print(f"Looking for input files in: {DATA_PROCESSED_DIR}")

    # 1. Load Manually Collected SEC Fee Changes
    manual_fees_path = DATA_PROCESSED_DIR / MANUAL_FEES_FILENAME
    if not manual_fees_path.exists():
        print(f"ERROR: Manual SEC fees file not found at {manual_fees_path}")
        print(
            f"Please ensure '{MANUAL_FEES_FILENAME}' is in the '{DATA_PROCESSED_DIR}' directory."
        )
        return

    fee_df = pd.read_csv(manual_fees_path)
    print(f"\nLoaded manual SEC fee data from: {manual_fees_path}")

    # Validate and prepare fee_df
    if not {"EffectiveDate", "RatePerMillion"}.issubset(fee_df.columns):
        print(
            f"ERROR: '{MANUAL_FEES_FILENAME}' must contain 'EffectiveDate' and 'RatePerMillion' columns."
        )
        return

    fee_df["EffectiveDate"] = pd.to_datetime(fee_df["EffectiveDate"], errors="coerce")
    fee_df["RatePerMillion"] = pd.to_numeric(fee_df["RatePerMillion"], errors="coerce")

    # Drop rows where essential data couldn't be parsed
    fee_df.dropna(subset=["EffectiveDate", "RatePerMillion"], inplace=True)

    fee_df = fee_df.sort_values("EffectiveDate").reset_index(drop=True)
    # Ensure no duplicate effective dates in the fee file, keep the last if any (though ideally none)
    fee_df = fee_df.drop_duplicates(subset=["EffectiveDate"], keep="last")

    print("\nProcessed SEC Fee Rate Changes (first 5 rows):")
    print(fee_df.head())
    print(f"Total fee change events loaded: {len(fee_df)}")

    # 2. Load the SPX Daily IV Panel
    iv_panel_path = DATA_PROCESSED_DIR / IV_PANEL_FILENAME
    if not iv_panel_path.exists():
        print(f"ERROR: SPX IV panel file not found at {iv_panel_path}")
        return

    panel_df = pd.read_csv(iv_panel_path)
    print(
        f"\nLoaded SPX daily IV panel from: {iv_panel_path} with {len(panel_df)} rows."
    )

    if "Date" not in panel_df.columns:
        print("ERROR: The IV panel must contain a 'Date' column.")
        return
    panel_df["Date"] = pd.to_datetime(panel_df["Date"], errors="coerce")
    panel_df.dropna(subset=["Date"], inplace=True)
    panel_df = panel_df.sort_values("Date").reset_index(drop=True)

    # Get first_fee_date_in_manual_file from the original fee_df before it's used in merge_asof
    first_fee_date_in_manual_file = None
    if (
        not fee_df.empty
        and "EffectiveDate" in fee_df.columns
        and pd.api.types.is_datetime64_any_dtype(fee_df["EffectiveDate"])
    ):
        valid_dates_in_fee_df = fee_df["EffectiveDate"].dropna()
        if not valid_dates_in_fee_df.empty:
            first_fee_date_in_manual_file = valid_dates_in_fee_df.min()

    # 3. Merge Fee Rate Data with the Panel
    merged_df = pd.merge_asof(
        left=panel_df,
        right=fee_df.rename(
            columns={"EffectiveDate": "Date", "RatePerMillion": "sec_fee_rate"}
        ),
        on="Date",
        direction="backward",
    )
    print("\nMerged SEC fee rates into the panel.")

    initial_na_count = merged_df["sec_fee_rate"].isnull().sum()
    if initial_na_count > 0:
        if first_fee_date_in_manual_file is not None:
            print(
                f"  Note: {initial_na_count} rows (likely at the beginning of the panel, before {first_fee_date_in_manual_file.strftime('%Y-%m-%d')}) have no preceding SEC fee rate in your manual file and will have NaN for 'sec_fee_rate'."
            )
        else:
            print(
                f"  Note: {initial_na_count} rows have NaN for 'sec_fee_rate'. This might be due to dates in the panel preceding all dates in the manual fee file, or issues with the fee file (e.g., all dates failed to parse)."
            )

    # 4. Identify Fee Change Events and Magnitudes
    merged_df["prev_sec_fee_rate"] = merged_df["sec_fee_rate"].shift(1)
    merged_df["fee_change_event"] = 0
    condition_change = (
        merged_df["sec_fee_rate"] != merged_df["prev_sec_fee_rate"]
    ) & merged_df["sec_fee_rate"].notna()
    condition_first_fee = (
        merged_df["prev_sec_fee_rate"].isna() & merged_df["sec_fee_rate"].notna()
    )

    merged_df.loc[condition_change | condition_first_fee, "fee_change_event"] = 1

    merged_df["fee_change_magnitude"] = 0.0
    merged_df.loc[
        condition_change & merged_df["prev_sec_fee_rate"].notna(),
        "fee_change_magnitude",
    ] = (
        merged_df["sec_fee_rate"] - merged_df["prev_sec_fee_rate"]
    )
    merged_df.loc[condition_first_fee, "fee_change_magnitude"] = merged_df[
        "sec_fee_rate"
    ]

    del merged_df["prev_sec_fee_rate"]
    print("Calculated fee change events and magnitudes.")

    # 5. Save the Augmented DataFrame
    output_file_path = DATA_PROCESSED_DIR / OUTPUT_PANEL_FILENAME
    merged_df.to_csv(output_file_path, index=False, date_format="%Y-%m-%d")
    print(f"\nSuccessfully processed data and saved to {output_file_path}")

    print(
        "\nPreview of data with fee information around a known change date (if available in your data):"
    )
    event_days_df = merged_df[merged_df["fee_change_event"] == 1].copy()

    if not event_days_df.empty:
        sample_event_date_for_preview = event_days_df["Date"].iloc[
            0
        ]  # Default to the first event

        # If there are multiple events, and the first event in event_days_df is also the
        # first day a fee rate appears in merged_df, try to pick the second event for a more interesting preview.
        if len(event_days_df) > 1:
            first_fee_application_index_in_merged_df = merged_df[
                "sec_fee_rate"
            ].first_valid_index()
            # event_days_df.index contains the original indices from merged_df for these event rows
            if (
                first_fee_application_index_in_merged_df is not None
                and event_days_df.index[0] == first_fee_application_index_in_merged_df
            ):
                sample_event_date_for_preview = event_days_df["Date"].iloc[1]

        # Get the integer index location (iloc) of this sample event date in merged_df
        sample_event_iloc_list = merged_df.index[
            merged_df["Date"] == sample_event_date_for_preview
        ].tolist()

        if sample_event_iloc_list:
            sample_event_iloc = sample_event_iloc_list[0]

            preview_start_iloc = max(0, sample_event_iloc - 3)
            preview_end_iloc = min(len(merged_df) - 1, sample_event_iloc + 3)

            preview_df = merged_df.iloc[preview_start_iloc : preview_end_iloc + 1]
            print(
                preview_df[
                    [
                        "Date",
                        "ATM_IV_1545",
                        "sec_fee_rate",
                        "fee_change_event",
                        "fee_change_magnitude",
                    ]
                ].to_string()
            )
        else:
            print(
                f"Could not find sample event date {sample_event_date_for_preview} in merged_df for preview (this should not happen if event_days_df is not empty)."
            )
    else:
        print(
            "No fee change events found in the merged data to show a preview around an event."
        )

    print("\nSummary of fee change events found in the merged panel:")
    event_summary = merged_df[merged_df["fee_change_event"] == 1][
        ["Date", "sec_fee_rate", "fee_change_magnitude"]
    ]
    print(event_summary.to_string())
    print(f"Total fee change event days identified in panel: {len(event_summary)}")


if __name__ == "__main__":
    if not DATA_PROCESSED_DIR.exists():
        print(f"Creating directory: {DATA_PROCESSED_DIR}")
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prepare_causal_data()
