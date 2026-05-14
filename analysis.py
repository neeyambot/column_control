# Author: Nazia
# Date: May 2026
# Goal: EDA and cleaning for distillation column control model 
import pandas as pd
import numpy as np

# Load data
df = pd.read_parquet("column_ml_dataset.parquet")
print(f"Shape: {df.shape}")

# Step 1: Find and drop constant columns automatically
constant_cols = []
for col in df.columns:
    if df[col].nunique() == 1:
        print(f"{col}: {df[col].nunique()} unique values → {sorted(df[col].dropna().unique())}")
        constant_cols.append(col)

print(f"\nAuto-detected constant columns: {constant_cols}")


# Checking if output_residue_molar_flow is constant. It seemed constant when I checked manually
print(df["output_residue_molar_flow"].nunique())
print(df["output_residue_molar_flow"].unique())

# Checking if state_DC_reflux_ratio is constant. It seemed constant when I checked manually
print(df["state_DC_reflux_ratio"].nunique())
print(df["state_DC_reflux_ratio"].unique())

# Prove duplicate columns are identical
diff1 = df["state_DC_condenser_duty"] - df["output_C_Duty_energy_flow"]
diff2 = df["state_DC_reboiler_duty"].abs() - df["output_R_Duty_energy_flow"]

print("state_DC_condenser_duty vs output_C_Duty_energy_flow:")
print(f"  Max difference: {diff1.abs().max()}")
print(f"  Both are equal: {(diff1 == 0).all()}")

print("\nstate_DC_reboiler_duty vs output_R_Duty_energy_flow:")
print(f"  Max difference: {diff2.abs().max()}")
print(f"  Both are equal: {(diff2 == 0).all()}")



# Step 2: Also drop redundant columns manually
REDUNDANT_COLS = [
    "sample_index",                # just a row number
    "timestamp_utc",               # not useful for modeling
    "input_feed_mass_flow",        # same info as molar flow
    "output_distillate_z_toluene", # toluene = 1 - benzene
    "output_residue_z_benzene",    # same reason
    "state_DC_condenser_duty",     # duplicate of output_C_Duty
    "state_DC_reboiler_duty",      # duplicate of output_R_Duty
    "output_residue_molar_flow",   #verified above, it is constant(5.8) except at 6 places where it is 5.8 when rounded
    "state_DC_reflux_ratio",        # same reason...
    ]
# Combine both lists and only drop columns that exist
DROP_COLS = constant_cols + [c for c in REDUNDANT_COLS if c in df.columns]

df_clean = df.drop(columns=DROP_COLS)

print(f"\nDropped {len(DROP_COLS)} columns total:")
for col in DROP_COLS:
    print(f"  - {col}")

print(f"\nShape before: {df.shape}")
print(f"Shape after:  {df_clean.shape}")

# Step 3: Save cleaned data
df_clean.to_csv("column_ml_clean.csv", index=False)
print(f"\nSaved → column_ml_clean.csv")
print(f"Final shape: {df_clean.shape}")
print(f"\nRemaining columns:\n{list(df_clean.columns)}")

