
# UK Used Car Adverts — Exploratory Data Analysis

Exploratory data analysis of **402,005 UK vehicle adverts**, covering data quality assessment, cleaning, feature engineering and the relationships between price, mileage, vehicle age and categorical attributes such as make, body type and fuel type.

**Notebook:** [`Car_project.ipynb`](Car_project.ipynb) 

---

## Dataset

`adverts.csv` — 402,005 rows × 12 columns.

| Column | Description |
|---|---|
| `public_reference` | Unique advert reference (used as the index) |
| `mileage` | Miles driven |
| `reg_code` | UK registration code |
| `standard_colour` | Vehicle colour |
| `standard_make` | Manufacturer (110 unique) |
| `standard_model` | Model (1,168 unique) |
| `vehicle_condition` | `NEW` or `USED` |
| `year_of_registration` | Year first registered |
| `price` | Advertised price (£) |
| `body_type` | Hatchback, SUV, saloon, etc. |
| `crossover_car_and_van` | Boolean flag |
| `fuel_type` | Petrol, diesel, hybrid, electric, etc. |

The CSV is not included in this repository. Place it at the path referenced in the first cells of the notebook before running.

## Data quality assessment

| Column | Missing values |
|---|---|
| `year_of_registration` | 33,311 |
| `reg_code` | 31,857 |
| `standard_colour` | 5,378 |
| `body_type` | 837 |
| `fuel_type` | 601 |
| `mileage` | 127 |

Auditing the used-vehicle subset separately showed only 2,062 missing registration years, confirming that the bulk of the missingness belongs to new vehicles — a structural pattern rather than a random one, which shaped the imputation strategy.

The raw data also contained implausible values: registration years such as `999`, `1015` and `1515`; mileage up to 999,999; and prices ranging from £120 to £9,999,999.

## Cleaning and preparation

- `public_reference` set as the DataFrame index.
- `year_of_registration` filled with 2020 and `reg_code` with 70 where absent, then remaining numeric gaps mean-imputed and categorical gaps mode-imputed (`SimpleImputer`).
- Implausible registration years removed (16 rows below 1800).
- Mileage capped at 400,000 and price at 300,000, with thresholds chosen after inspecting box and strip plots; a further price filter below £40,000 used for distribution plots.
- Final working dataset: **376,486 complete records**.

## Feature engineering

`vehicle_age = 2024 − year_of_registration`, set to 0 for vehicles with condition `NEW`. Mean 8.5 years, median 8, interquartile range 6–11.

## Key findings

**Correlations among numeric features**

|  | mileage | year_of_registration | price |
|---|---|---|---|
| **mileage** | 1.000 | −0.727 | −0.548 |
| **year_of_registration** | −0.727 | 1.000 | 0.569 |
| **price** | −0.548 | 0.569 | 1.000 |

- Newer vehicles carry substantially lower mileage, and mileage is the strongest negative driver of price in this dataset.
- Price rises with registration year, with the steepest depreciation in the first few years and a flattening after roughly ten years.
- Premium marques (Mercedes-Benz, BMW, Land Rover, Audi) show higher median prices and much wider price spreads than volume brands.
- The market is dominated by used stock: 370,756 of 402,005 adverts are `USED`. BMW is the most advertised make (37,376 adverts), Golf the most advertised model (11,583), and black the most common colour (86,287).
- Hatchbacks and SUVs dominate body types, with petrol and diesel accounting for the overwhelming majority of fuel types; electric and hybrid listings are a small minority.

## Tech stack

Python · pandas · NumPy · scikit-learn (`SimpleImputer`) · Matplotlib · Seaborn · Plotly

## Running the notebook

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly
jupyter notebook Car_project.ipynb
```

The notebook was originally written in Google Colab and mounts Google Drive to load the CSV. To run locally, replace the `drive.mount(...)` cell and point `pd.read_csv(...)` at your local copy of `adverts.csv`.

## Next steps

The cleaned dataset and engineered age feature provide a ready foundation for predictive modelling — a gradient-boosted regressor over mileage, vehicle age, make, model, body type and fuel type, with SHAP used to decompose feature contributions to predicted price.
