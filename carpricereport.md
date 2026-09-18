# Exploratory Data Analysis of UK Vehicle Adverts

**Author:** Taniya Gidatkar
**Dataset:** `adverts.csv` — 402,005 vehicle adverts, 12 attributes
**Notebook:** `Car_project.ipynb`

---

## 1. Objective

This project explores a large set of UK vehicle adverts to understand the structure of the market and identify which attributes are associated with advertised price. The work covers data profiling, missing-value treatment, outlier handling, feature engineering and a full programme of visual analysis across quantitative–quantitative, quantitative–categorical and categorical–categorical relationships.

## 2. Data profiling

The analysis began by loading the dataset and establishing its shape and composition: 402,005 rows across 12 columns. Using `select_dtypes`, the columns were separated into four quantitative features (`public_reference`, `mileage`, `year_of_registration`, `price`) and eight qualitative features (`reg_code`, `standard_colour`, `standard_make`, `standard_model`, `vehicle_condition`, `body_type`, `crossover_car_and_van`, `fuel_type`).

`public_reference` was identified as an advert identifier rather than a measurement and was set as the DataFrame index, removing it from the numeric feature space where it would otherwise distort summary statistics and correlations.

Summary statistics for the numeric columns were produced with `describe()`:

| Statistic | mileage | year_of_registration | price (£) |
|---|---|---|---|
| count | 401,878 | 368,694 | 402,005 |
| mean | 37,744 | 2015.0 | 17,342 |
| std | 34,832 | 7.96 | 46,437 |
| min | 0 | 999 | 120 |
| 25% | 10,481 | 2013 | 7,495 |
| 50% | 28,630 | 2016 | 12,600 |
| 75% | 56,876 | 2018 | 20,000 |
| max | 999,999 | 2020 | 9,999,999 |

Profiling the categorical columns established the market composition: 110 manufacturers and 1,168 models, with BMW the most advertised make (37,376 adverts), the Volkswagen Golf the most advertised model (11,583) and black the most common colour (86,287). The market is overwhelmingly second-hand, with 370,756 of 402,005 adverts listed as `USED`.

## 3. Data quality assessment

### 3.1 Missing values

A null audit across all columns quantified the gaps:

| Column | Missing | % of rows |
|---|---|---|
| `year_of_registration` | 33,311 | 8.3% |
| `reg_code` | 31,857 | 7.9% |
| `standard_colour` | 5,378 | 1.3% |
| `body_type` | 837 | 0.2% |
| `fuel_type` | 601 | 0.1% |
| `mileage` | 127 | <0.1% |

To understand *why* the two largest gaps existed, the used-vehicle subset was isolated and audited separately. Within used vehicles only 2,062 registration years and 608 registration codes were missing — demonstrating that the missingness is concentrated in new vehicles, which by definition have not yet been registered. This diagnostic step turned an apparently random 8% gap into an explainable, structural one and directly informed the imputation strategy that followed.

### 3.2 Outliers

An interquartile-range test (values beyond 1.5 × IQR from the quartiles) was applied to every numeric column, flagging 8,181 mileage outliers, 11,915 registration-year outliers and 26,269 price outliers. Because genuine high-value and high-mileage vehicles exist in this market, these flags were treated as candidates for inspection rather than automatic deletion, and each column was then examined visually before any thresholds were set.

## 4. Cleaning

Cleaning was carried out in a deliberate sequence:

1. `public_reference` set as the index.
2. `year_of_registration` filled with 2020 and `reg_code` with 70 for the new-vehicle cohort identified in the diagnostic above.
3. Remaining numeric gaps mean-imputed and categorical gaps mode-imputed using scikit-learn's `SimpleImputer`, then re-audited to confirm a complete dataset.
4. Implausible registration years removed — 16 records carrying values such as 999, 1015 and 1515.
5. Mileage capped at 400,000 and price at 300,000, with thresholds selected after inspecting box plots and strip plots of each distribution.

A final null check confirmed zero missing values across all columns, leaving 376,486 complete records.

## 5. Distribution analysis

Price was examined with a KDE plot and a histogram over the sub-£40,000 segment, where the bulk of the market sits. The distribution is clearly right-skewed, with a mean of £13,639, a median of £11,995 and an interquartile range of £7,150 to £18,490 — the classic long-tail shape of a consumer vehicle market.

Mileage was investigated with a combined box plot and strip plot, overlaying individual observations on the summary statistics. The first pass showed the extreme tail compressing all meaningful variation into a narrow band, so the view was re-plotted at the 80th percentile and then at the 400,000-mile cap, producing a legible distribution in which the central mass and the tail could both be read. This iterative approach — plot, diagnose, adjust the view, re-plot — was applied throughout the visual analysis.

## 6. Feature engineering

A `vehicle_age` feature was derived as `2024 − year_of_registration`, with age explicitly set to 0 for records where `vehicle_condition` is `NEW`. The resulting distribution has a mean of 8.5 years, a median of 8 years and an interquartile range of 6 to 11 years, with an upper tail extending to 115 years representing classic and vintage vehicles. This single engineered feature made depreciation directly analysable in a way that raw registration year did not.

## 7. Relationship analysis

### 7.1 Quantitative–quantitative

A correlation matrix and scatter-plot matrix were produced across mileage, registration year and price:

|  | mileage | year_of_registration | price |
|---|---|---|---|
| **mileage** | 1.000 | −0.727 | −0.548 |
| **year_of_registration** | −0.727 | 1.000 | 0.569 |
| **price** | −0.548 | 0.569 | 1.000 |

Mileage and registration year are strongly negatively correlated at −0.73, confirming that older vehicles have accumulated more distance. Both relate to price in the expected directions: price falls as mileage rises (−0.55) and rises with registration year (0.57). Mileage emerges as the single strongest numeric driver of price in the dataset.

A grouped bar chart of mean and minimum price against vehicle age made the depreciation curve explicit — the steepest decline occurs in the first few years, flattening considerably after roughly ten years, with a slight uplift at the extreme age end where classic vehicles appreciate rather than depreciate. A point plot of average mileage by registration year traced a near-monotonic decline from older to newer cohorts, and a bar plot of mean mileage by year confirmed the same pattern.

### 7.2 Quantitative–categorical

Price distribution across the ten most-advertised makes was examined with box plots. The separation is clear: premium marques such as Mercedes-Benz, BMW, Land Rover and Audi carry higher median prices and noticeably wider interquartile ranges, reflecting model line-ups that span entry-level to flagship vehicles. Volume brands including Vauxhall, Ford and Nissan cluster at lower medians with tighter spreads. Manufacturer is therefore a meaningful signal for price.

A grouped count of vehicle condition by make quantified how new and used stock is distributed across manufacturers.

### 7.3 Categorical–categorical

The ten most common body types were cross-tabulated against fuel type using count plots. Hatchbacks and SUVs dominate by volume, and petrol and diesel account for almost all listings within every body type. Electric and hybrid vehicles form a small share concentrated in hatchbacks and SUVs — consistent with a dataset captured in 2020, ahead of the sharpest growth in UK electric vehicle adoption.

Colour was analysed against the top ten makes with a horizontal count plot. Black, grey, white, blue and silver dominate consistently across every manufacturer, showing that colour preference is a market-wide pattern rather than a brand-specific one.

Manufacturer frequency across the whole dataset was also visualised with an interactive Plotly histogram, and fuel-type composition with a count plot, giving an overall picture of market share before the more targeted comparisons above.

## 8. Conclusions

The analysis establishes a clear and internally consistent picture of the UK vehicle advert market. Advertised price is driven principally by vehicle age and mileage, moderated by manufacturer and body type, and the depreciation curve recovered from the data matches established behaviour in the used-car market. The dataset was profiled, diagnosed, cleaned to completeness and enriched with an engineered age feature, leaving 376,486 analysis-ready records and a set of validated relationships suitable for downstream predictive modelling.

## 9. Tools

Python · pandas · NumPy · scikit-learn (`SimpleImputer`) · Matplotlib · Seaborn · Plotly
