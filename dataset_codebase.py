import pandas as pd
import re

# Set display option for Pandas
pd.options.display.max_rows = 50

# Load US company universe Excel file, skipping header rows and keeping only relevant columns
df_companies_universe = (
    pd.read_excel(
        "US Company Universe.xlsx",
        skiprows=3,
        header=0,
        dtype={"SP_ENTITY_ID": str},
    )
    .dropna(how="all")
    .reset_index(drop=True)
)

# Keep only necessary columns and drop duplicate company names
columns_to_keep = [
    "SP_ENTITY_NAME", "SP_ENTITY_ID", "SP_COMPANY_NAME", "SP_COMPANY_STATUS",
    "IQ_INDUSTRY_CLASSIFICATION", "SP_COMPANY_TYPE", "SPONSOR_BACKED_INFO",
]
df_companies_universe = df_companies_universe[columns_to_keep].drop_duplicates(
    subset="SP_COMPANY_NAME", keep=False
)

# Load RepRisk ID mapping CSV and drop duplicate company names
df_reprisk_id = pd.read_csv("RepRisk Identifiers.csv")
print(df_reprisk_id.shape)
df_reprisk_id = df_reprisk_id.drop_duplicates(subset="company_name", keep=False)
print(df_reprisk_id.shape)

# Function to clean company names: lowercase, remove punctuation, collapse spaces
def clean_name(name):
    if isinstance(name, str):
        name = name.lower()
        name = re.sub(r"[^\w\s]", "", name)
        name = re.sub(r"\s+", " ", name)
        return name.strip()
    return ""

# Clean company names in both datasets
df_companies_universe["SP_COMPANY_NAME_CLEAN"] = df_companies_universe["SP_COMPANY_NAME"].map(clean_name)
df_reprisk_id["company_name_clean"] = df_reprisk_id["company_name"].map(clean_name)

# Drop duplicates again based on cleaned names
df_companies_universe = df_companies_universe.drop_duplicates(subset="SP_COMPANY_NAME_CLEAN", keep="first")
df_reprisk_id = df_reprisk_id.drop_duplicates(subset="company_name_clean", keep="first")

# Merge company universe with RepRisk ID mapping using cleaned names
df_comp_repiq = df_companies_universe.merge(
    df_reprisk_id,
    left_on="SP_COMPANY_NAME_CLEAN",
    right_on="company_name_clean",
    how="inner",
    validate="one_to_one",
)

# Drop unnecessary columns after merge and rename for clarity
columns_to_drop = [
    "headquarters_country_isocode", "url", "isins", "primary_isin", "company_name_clean",
    "headquarters_country", "IQ_INDUSTRY_CLASSIFICATION", "SP_COMPANY_NAME_CLEAN",
]
df_comp_repiq = df_comp_repiq.drop(columns=columns_to_drop).rename(
    columns={"company_name": "company_name_reprisk"}
)

# Load PE firm entry-exit data and filter for relevant US PE cases
df_entry_exit = pd.read_excel("PE_Firm_Entry_Exit.xlsx", dtype={"Target ID": str})
df_entry_exit = df_entry_exit[df_entry_exit["USPE"] == 1]
df_entry_exit = df_entry_exit[df_entry_exit["Target ID"].notna()]

# Convert entry and exit dates to datetime
df_entry_exit["Entry_Date"] = pd.to_datetime(df_entry_exit["Entry_Date"], errors="coerce")
df_entry_exit["Exit_Date"] = pd.to_datetime(df_entry_exit["Exit_Date"], errors="coerce")

# Determine PE backing status for each year between 2007–2020
years = list(range(2007, 2021))
results = []

for target_id, group in df_entry_exit.groupby("Target ID"):
    group["Latest_Date"] = group[["Entry_Date", "Exit_Date"]].max(axis=1)
    latest_row = group.iloc[-1, :] if group["Latest_Date"].isna().all() else group.loc[group["Latest_Date"].idxmax()]
    company = latest_row["Company"]
    target = latest_row["Target"]
    backed_status = {year: [] for year in years}

    for _, row in group.iterrows():
        entry = row["Entry_Date"] or pd.Timestamp("2007-01-01")
        exit = row["Exit_Date"] or pd.Timestamp("2020-12-31")
        if exit.year < 2007:
            continue
        for year in years:
            if entry.year <= year <= exit.year:
                if row["PE_Firm"] not in backed_status[year]:
                    backed_status[year].append(row["PE_Firm"])

    for year, firms in backed_status.items():
        results.append({
            "Company": company,
            "Target ID": target_id,
            "Target": target,
            "Year": year,
            "PE_Firm": firms,
            "PE_backed": bool(firms),
        })

# Build PE backing status dataframe by year
df_PE_by_year = pd.DataFrame(results, copy=True)

# Create all company-year combinations for RepRisk-PE data
df_comp_repiq_year = df_comp_repiq.merge(pd.DataFrame({"Year": years}), how="cross")

# Merge PE status into RepRisk-matched dataset
df_w_PE_status = df_comp_repiq_year.merge(
    df_PE_by_year,
    how="left",
    right_on=["Target ID", "Year"],
    left_on=["SP_ENTITY_ID", "Year"],
    validate="one_to_one",
).drop(columns=["Company", "Target ID"])

# Fill missing values and format columns
df_w_PE_status["PE_Firm"] = df_w_PE_status["PE_Firm"].apply(lambda x: x if isinstance(x, list) else [])
df_w_PE_status["PE_backed"] = df_w_PE_status["PE_backed"].fillna(False)

# Load RepRisk index scores and clean
df_reprisk_index = pd.read_csv("RepRisk Index.csv")
columns_to_drop = ["name", "headquarter_country", "url", "all_ISINs", "primary_ISIN"]
df_reprisk_index = df_reprisk_index.drop(columns=columns_to_drop)

# Convert date columns and percentages to numeric format
df_reprisk_index["date"] = pd.to_datetime(df_reprisk_index["date"])
df_reprisk_index["peak_RRI_date"] = pd.to_datetime(df_reprisk_index["peak_RRI_date"], errors="coerce")
df_reprisk_index["year"] = df_reprisk_index["date"].dt.year
df_reprisk_index = df_reprisk_index.sort_values(["RepRisk_ID", "date"])

for col in ["environmental_percentage", "social_percentage", "governance_percentage"]:
    df_reprisk_index[col] = df_reprisk_index[col].str.replace("%", "").astype(float)

# Custom mode function for categorical ESG ratings
rating_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']
def custom_mode(series):
    if series.dropna().empty:
        return None
    mode_values = series.mode()
    if mode_values.empty:
        return None
    if len(mode_values) == 1:
        return mode_values.iloc[0]
    ranked = sorted(mode_values, key=lambda x: rating_order.index(x))
    return ranked[-1]

# Aggregate RepRisk metrics at yearly level
df_reprisk_yearly = (
    df_reprisk_index.groupby(["RepRisk_ID", "year"])
    .agg({
        "current_RRI": "mean", "RRI_trend": "sum", "peak_RRI": "max",
        "peak_RRI_date": "max", "RepRisk_rating": custom_mode,
        "country_sector_average": "mean", "environmental_percentage": "mean",
        "social_percentage": "mean", "governance_percentage": "mean",
        "headquarter_country_code": "first", "sectors": "first",
    })
    .reset_index()
)

# Merge RepRisk yearly scores into combined dataset
df_semi_final = pd.merge(
    df_w_PE_status,
    df_reprisk_yearly,
    how='left',
    left_on=['reprisk_id', 'Year'],
    right_on=['RepRisk_ID', 'year']
).drop(columns=["RepRisk_ID", "Year", "company_name_reprisk"])

# Load ESG incidents data and clean
df_incidents = pd.read_csv("RepRisk Incidents.csv")
df_incidents.drop(columns=["headquarters_country_isocode", "primary_isin", "story_id"], inplace=True)

# Identify boolean columns with T/F values
bool_cols = df_incidents.isin(["T", "F"]).any().pipe(lambda s: s[s].index.tolist())

# Process incident data: convert types, group by year + company
incidents_for_final_df = (
    df_incidents
    .assign(
        incident_date=lambda d: pd.to_datetime(d["incident_date"]),
        year=lambda d: d.incident_date.dt.year
    )
    .replace({"T": True, "F": False})
    .groupby(["year", "reprisk_id"])
    .agg(
        severity=("severity", "mean"),
        reach=("reach", "mean"),
        novelty=("novelty", "mean"),
        **{c: (c, "sum") for c in bool_cols},
        unsharp_incident=("unsharp_incident", "sum"),
        related_countries=("related_countries", lambda x: x.dropna().unique().tolist()),
        related_countries_codes=("related_countries_codes", lambda x: x.dropna().unique().tolist()),
        no_incidents=("reprisk_id", "size")
    )
    .reset_index()
    .sort_values(["reprisk_id", "year"])
)

# Final dataset: merge all ESG scores and incident data
df_final = pd.merge(
    df_semi_final,
    incidents_for_final_df,
    how='left',
    on=['reprisk_id', 'year']
)

df_final.to_csv("final_dataset.csv")