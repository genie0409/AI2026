import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Matplotlib Korean Font Settings (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 2. File Paths
data_dir = r"c:\Users\user\Desktop\workspace\data"
path_sat = os.path.join(data_dir, "삶의_만족도_시도__20260606195059.xlsx")
path_sui = os.path.join(data_dir, "인구십만명당_자살률_시도_시_군_구__20260606194913.xlsx")

# Output directory for artifacts
artifact_dir = r"C:\Users\user\.gemini\antigravity\brain\9cd79b94-8971-4bc6-aecc-aaa399e554bd"
os.makedirs(artifact_dir, exist_ok=True)

# ----------------------------------------------------
# 3. Load & Process Life Satisfaction Data
# ----------------------------------------------------
print("Processing Life Satisfaction Data...")
df_sat_raw = pd.read_excel(path_sat, sheet_name="데이터")

# Forward fill regions (column 0) and gender_type (column 1)
df_sat = df_sat_raw.copy()
df_sat.iloc[:, 0] = df_sat.iloc[:, 0].ffill()
df_sat.iloc[:, 1] = df_sat.iloc[:, 1].ffill()

# Set up column mapping
# Columns 0, 1, 2 are region, gender_type, gender
# Row 0 contains sub-headers: 계, 매우 만족, 약간 만족, 보통, 약간 불만족, 매우 불만족
sub_headers = df_sat.iloc[0].values

sat_records = []
for idx in range(1, len(df_sat)):
    row = df_sat.iloc[idx]
    region = str(row.iloc[0]).strip()
    gender_type = str(row.iloc[1]).strip()
    gender_val = str(row.iloc[2]).strip()
    
    # Standardize gender mapping
    if gender_type == "전체" and gender_val == "계":
        gender = "계"
    elif gender_type == "성별":
        gender = gender_val
    else:
        continue  # skip other groupings if any
        
    for col_idx in range(3, len(df_sat.columns)):
        col_name = df_sat.columns[col_idx]
        # Extract year
        try:
            year = int(float(col_name))
        except ValueError:
            continue
            
        metric = sub_headers[col_idx]
        val = row.iloc[col_idx]
        
        # Convert value to float
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            val_float = np.nan
            
        sat_records.append({
            "region": region,
            "gender": gender,
            "year": year,
            "metric": metric,
            "value": val_float
        })

df_sat_clean = pd.DataFrame(sat_records)
# Pivot to wide format: columns for each metric
df_sat_wide = df_sat_clean.pivot(index=["region", "gender", "year"], columns="metric", values="value").reset_index()

# Rename columns for clarity
df_sat_wide = df_sat_wide.rename(columns={
    "계": "sat_total",
    "매우 만족": "sat_very_satisfied",
    "약간 만족": "sat_somewhat_satisfied",
    "보통": "sat_neutral",
    "약간 불만족": "sat_somewhat_dissatisfied",
    "매우 불만족": "sat_very_dissatisfied"
})

# Calculate Satisfaction Index and Average Score
# Satisfaction score (1-5 scale)
# score = (1*very_dissatisfied + 2*somewhat_dissatisfied + 3*neutral + 4*somewhat_satisfied + 5*very_satisfied) / 100
df_sat_wide["satisfaction_score"] = (
    1 * df_sat_wide["sat_very_dissatisfied"].fillna(0) +
    2 * df_sat_wide["sat_somewhat_dissatisfied"].fillna(0) +
    3 * df_sat_wide["sat_neutral"].fillna(0) +
    4 * df_sat_wide["sat_somewhat_satisfied"].fillna(0) +
    5 * df_sat_wide["sat_very_satisfied"].fillna(0)
) / 100.0

# If all satisfaction columns are NaN, set score to NaN
nan_mask = df_sat_wide[["sat_very_satisfied", "sat_somewhat_satisfied", "sat_neutral", "sat_somewhat_dissatisfied", "sat_very_dissatisfied"]].isna().all(axis=1)
df_sat_wide.loc[nan_mask, "satisfaction_score"] = np.nan

# Positive rate (매우 만족 + 약간 만족)
df_sat_wide["satisfaction_positive_rate"] = df_sat_wide["sat_very_satisfied"].fillna(0) + df_sat_wide["sat_somewhat_satisfied"].fillna(0)
df_sat_wide.loc[nan_mask, "satisfaction_positive_rate"] = np.nan

# Negative rate (매우 불만족 + 약간 불만족)
df_sat_wide["satisfaction_negative_rate"] = df_sat_wide["sat_very_dissatisfied"].fillna(0) + df_sat_wide["sat_somewhat_dissatisfied"].fillna(0)
df_sat_wide.loc[nan_mask, "satisfaction_negative_rate"] = np.nan

print(f"Processed Satisfaction Data shape: {df_sat_wide.shape}")

# ----------------------------------------------------
# 4. Load & Process Suicide Rate Data
# ----------------------------------------------------
print("Processing Suicide Rate Data...")
df_sui_raw = pd.read_excel(path_sui, sheet_name="데이터")

df_sui = df_sui_raw.copy()
sub_headers_sui = df_sui.iloc[0].values

sui_records = []
for idx in range(1, len(df_sui)):
    row = df_sui.iloc[idx]
    region = str(row.iloc[0]).strip()
    
    for col_idx in range(1, len(df_sui.columns)):
        col_name = df_sui.columns[col_idx]
        try:
            year = int(float(col_name))
        except ValueError:
            continue
            
        gender_val = str(sub_headers_sui[col_idx]).strip()
        # Map gender values to match sat
        # sat has: 계, 남자, 여자
        # sui has: 계, 남자, 여자
        val = row.iloc[col_idx]
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            val_float = np.nan
            
        sui_records.append({
            "region": region,
            "gender": gender_val,
            "year": year,
            "suicide_rate": val_float
        })

df_sui_clean = pd.DataFrame(sui_records)
print(f"Processed Suicide Data shape: {df_sui_clean.shape}")

# ----------------------------------------------------
# 5. Merge Data
# ----------------------------------------------------
print("Merging Datasets...")
df_merged = pd.merge(df_sat_wide, df_sui_clean, on=["region", "gender", "year"], how="inner")
print(f"Merged Data shape: {df_merged.shape}")

# Clean region names to make sure they match and standard
# Standardize names if there's any difference
# E.g. '강원특별자치도' vs '강원도'
print("Merged Regions:", df_merged["region"].unique())

# Drop rows with NaNs in key variables
df_analysis = df_merged.dropna(subset=["satisfaction_score", "suicide_rate"])
print(f"Analysis Data shape (after dropna): {df_analysis.shape}")

# Save the cleaned dataset for reference
df_analysis.to_csv(os.path.join(artifact_dir, "cleaned_data.csv"), index=False, encoding="utf-8-sig")

# ----------------------------------------------------
# 6. Statistical Analysis (Correlation)
# ----------------------------------------------------
print("Performing Correlation Analysis...")

# Overall Correlation (all years, all regions, gender == '계')
df_total = df_analysis[df_analysis["gender"] == "계"]
pearson_r_score, p_val_score = stats.pearsonr(df_total["satisfaction_score"], df_total["suicide_rate"])
spearman_r_score, sp_p_val_score = stats.spearmanr(df_total["satisfaction_score"], df_total["suicide_rate"])

pearson_r_pos, p_val_pos = stats.pearsonr(df_total["satisfaction_positive_rate"], df_total["suicide_rate"])
pearson_r_neg, p_val_neg = stats.pearsonr(df_total["satisfaction_negative_rate"], df_total["suicide_rate"])

print(f"Overall (Gender=Total) Satisfaction Score vs Suicide Rate Pearson r: {pearson_r_score:.4f} (p-val: {p_val_score:.4e})")
print(f"Overall (Gender=Total) Satisfaction Score vs Suicide Rate Spearman r: {spearman_r_score:.4f}")
print(f"Overall (Gender=Total) Positive Rate vs Suicide Rate Pearson r: {pearson_r_pos:.4f}")
print(f"Overall (Gender=Total) Negative Rate vs Suicide Rate Pearson r: {pearson_r_neg:.4f}")

# Gender-specific Correlation
gender_corrs = {}
for g in ["계", "남자", "female_dummy"]: # we have '남자', '여자', '계'
    # Wait, let's find unique genders
    pass

genders = df_analysis["gender"].unique()
print("Available genders:", genders)
for g in genders:
    df_g = df_analysis[df_analysis["gender"] == g]
    if len(df_g) > 2:
        r, p = stats.pearsonr(df_g["satisfaction_score"], df_g["suicide_rate"])
        r_sp, p_sp = stats.spearmanr(df_g["satisfaction_score"], df_g["suicide_rate"])
        gender_corrs[g] = {
            "pearson_r": r, "pearson_p": p,
            "spearman_r": r_sp, "spearman_p": p_sp
        }
        print(f"Gender {g} - Pearson r: {r:.4f} (p: {p:.4f}), Spearman r: {r_sp:.4f}")

# Year-specific Correlation (Gender = '계')
year_corrs = {}
years = sorted(df_total["year"].unique())
for y in years:
    df_y = df_total[df_total["year"] == y]
    # Filter out '전국' if we want regional variation only
    df_y_regions = df_y[df_y["region"] != "전국"]
    if len(df_y_regions) > 2:
        r, p = stats.pearsonr(df_y_regions["satisfaction_score"], df_y_regions["suicide_rate"])
        r_neg, p_neg = stats.pearsonr(df_y_regions["satisfaction_negative_rate"], df_y_regions["suicide_rate"])
        year_corrs[y] = {"pearson_r": r, "p_value": p, "r_neg": r_neg, "p_neg": p_neg}
        print(f"Year {y} (Regional variation) - Satisfaction Score Pearson r: {r:.4f} (p: {p:.4f})")
        print(f"Year {y} (Regional variation) - Negative Rate Pearson r: {r_neg:.4f} (p: {p_neg:.4f})")

# ----------------------------------------------------
# 7. Visualization Generation
# ----------------------------------------------------
print("Generating Plots...")

# Plot 1: correlation_scatter_by_gender.png
plt.figure(figsize=(10, 6))
# Exclude '전국' from scatter to show regional variation
df_scatter = df_analysis[df_analysis["region"] != "전국"]
colors = {"계": "#3498db", "남자": "#e74c3c", "여자": "#2ecc71"}
markers = {"계": "o", "남자": "^", "여자": "s"}

sns.scatterplot(
    data=df_scatter, 
    x="satisfaction_score", 
    y="suicide_rate", 
    hue="gender", 
    palette=colors,
    style="gender",
    markers=markers,
    alpha=0.8,
    s=80
)

# Fit regression lines for each gender
for g in genders:
    df_g = df_scatter[df_scatter["gender"] == g]
    sns.regplot(
        data=df_g,
        x="satisfaction_score",
        y="suicide_rate",
        scatter=False,
        label=f"{g} 회귀선 (r={gender_corrs[g]['pearson_r']:.2f})",
        color=colors[g],
        line_kws={"linestyle": "--" if g != "계" else "-"}
    )

plt.title("삶의 만족도 점수와 자살률 간의 상관관계 (성별 구분, 2020-2024)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("삶의 만족도 평균 점수 (5점 만점)", fontsize=12, labelpad=10)
plt.ylabel("인구 10만 명당 자살률 (명)", fontsize=12, labelpad=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(title="성별 구분", fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(artifact_dir, "correlation_scatter_by_gender.png"), dpi=300)
plt.close()

# Plot 2: trend_by_region.png
# Show National average (전국) and major regions (서울특별시, 부산광역시, 경기도, 강원특별자치도, 전라남도)
plt.figure(figsize=(12, 7))
major_regions = ["전국", "서울특별시", "부산광역시", "경기도", "강원특별자치도"]
df_trend = df_analysis[(df_analysis["gender"] == "계") & (df_analysis["region"].isin(major_regions))]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# Satisfaction Trend
sns.lineplot(
    data=df_trend,
    x="year",
    y="satisfaction_score",
    hue="region",
    marker="o",
    linewidth=2.5,
    ax=ax1,
    palette="Set1"
)
ax1.set_title("연도별 삶의 만족도 점수 추이", fontsize=12, fontweight="bold", pad=10)
ax1.set_xlabel("연도", fontsize=10)
ax1.set_ylabel("만족도 평균 점수 (5점 만점)", fontsize=10)
ax1.set_xticks(years)
ax1.grid(True, linestyle=":", alpha=0.6)

# Suicide Rate Trend
sns.lineplot(
    data=df_trend,
    x="year",
    y="suicide_rate",
    hue="region",
    marker="s",
    linewidth=2.5,
    ax=ax2,
    palette="Set1",
    legend=False # Legend is already on ax1
)
ax2.set_title("연도별 자살률 추이 (인구 10만 명당)", fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel("연도", fontsize=10)
ax2.set_ylabel("자살률 (명)", fontsize=10)
ax2.set_xticks(years)
ax2.grid(True, linestyle=":", alpha=0.6)

# Single legend for the figure
handles, labels = ax1.get_legend_handles_labels()
ax1.get_legend().remove()
fig.legend(handles, labels, loc="lower center", ncol=len(major_regions), bbox_to_anchor=(0.5, -0.05))

plt.suptitle("주요 지역별 삶의 만족도와 자살률 시계열 추이 (2020-2024)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(artifact_dir, "trend_by_region.png"), dpi=300, bbox_inches="tight")
plt.close()

# Plot 3: regional_comparison_bar.png
# 2024 data for all regions except '전국'
df_2024 = df_analysis[(df_analysis["year"] == 2024) & (df_analysis["gender"] == "계") & (df_analysis["region"] != "전국")]
df_2024 = df_2024.sort_values(by="suicide_rate", ascending=False)

fig, ax1 = plt.subplots(figsize=(14, 7))

# Bar chart for Suicide Rate (Left Y-axis)
color_bar = "#e74c3c"
bars = ax1.bar(df_2024["region"], df_2024["suicide_rate"], color=color_bar, alpha=0.7, label="자살률 (10만 명당)")
ax1.set_xlabel("행정구역", fontsize=12, labelpad=10)
ax1.set_ylabel("자살률 (명)", color=color_bar, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color_bar)
plt.xticks(rotation=45, ha='right')

# Line chart for Satisfaction Positive Rate (Right Y-axis)
ax2 = ax1.twinx()
color_line = "#2c3e50"
ax2.plot(df_2024["region"], df_2024["satisfaction_positive_rate"], color=color_line, marker="o", linewidth=2.5, label="삶의 만족도 긍정 비율 (%)")
ax2.set_ylabel("삶의 만족도 긍정 비율 (%)", color=color_line, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color_line)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, color=color_bar, fontweight="bold")

# Add values on line points
for i, txt in enumerate(df_2024["satisfaction_positive_rate"]):
    ax2.annotate(f'{txt:.1f}%', 
                 (df_2024["region"].iloc[i], df_2024["satisfaction_positive_rate"].iloc[i]),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center', fontsize=9, color=color_line, fontweight="bold")

plt.title("2024년 시도별 자살률과 삶의 만족도 긍정 비율 비교", fontsize=15, fontweight="bold", pad=15)
ax1.grid(True, axis='y', linestyle=":", alpha=0.5)

# Legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(artifact_dir, "regional_comparison_bar.png"), dpi=300)
plt.close()

# Plot 4: heatmap_satisfaction_suicide.png
# We want to see correlation between suicide_rate and:
# ['sat_very_satisfied', 'sat_somewhat_satisfied', 'satisfaction_positive_rate', 'sat_neutral', 'sat_somewhat_dissatisfied', 'sat_very_dissatisfied', 'satisfaction_negative_rate', 'satisfaction_score']
# For each gender
heatmap_data = []
for g in genders:
    df_g = df_analysis[(df_analysis["gender"] == g) & (df_analysis["region"] != "전국")]
    metrics = [
        "sat_very_satisfied", "sat_somewhat_satisfied", "satisfaction_positive_rate",
        "sat_neutral",
        "sat_somewhat_dissatisfied", "sat_very_dissatisfied", "satisfaction_negative_rate",
        "satisfaction_score"
    ]
    corrs = []
    for m in metrics:
        r, _ = stats.pearsonr(df_g[m], df_g["suicide_rate"])
        corrs.append(r)
    heatmap_data.append(corrs)

df_heatmap = pd.DataFrame(
    heatmap_data, 
    index=[f"성별: {g}" for g in genders], 
    columns=[
        "매우 만족", "약간 만족", "긍정 비율(합)", 
        "보통", 
        "약간 불만족", "매우 불만족", "부정 비율(합)", 
        "만족도 평균 점수"
    ]
)

plt.figure(figsize=(12, 5))
sns.heatmap(
    df_heatmap, 
    annot=True, 
    cmap="RdBu_r", 
    vmin=-1.0, 
    vmax=1.0, 
    fmt=".3f", 
    linewidths=.5,
    cbar_kws={'label': '피어슨 상관계수 (r)'}
)
plt.title("삶의 만족도 세부 지표와 자살률 간의 상관관계 히트맵 (시도별 데이터 기준)", fontsize=14, fontweight="bold", pad=15)
plt.xticks(rotation=15, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(artifact_dir, "heatmap_satisfaction_suicide.png"), dpi=300)
plt.close()

print("All plots generated successfully!")

# ----------------------------------------------------
# 8. Save text summary of analysis
# ----------------------------------------------------
summary_path = os.path.join(artifact_dir, "analysis_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=== 삶의 만족도와 자살률 상관관계 분석 결과 요약 ===\n\n")
    f.write("1. 전체 집단(성별: 계) 종합 상관계수 (2020-2024 전국/시도 포함)\n")
    f.write(f"  - 만족도 평균 점수 vs 자살률 Pearson r: {pearson_r_score:.4f} (p-value: {p_val_score:.4e})\n")
    f.write(f"  - 만족도 평균 점수 vs 자살률 Spearman r: {spearman_r_score:.4f} (p-value: {sp_p_val_score:.4e})\n")
    f.write(f"  - 긍정 비율(만족) vs 자살률 Pearson r: {pearson_r_pos:.4f} (p-value: {p_val_pos:.4e})\n")
    f.write(f"  - 부정 비율(불만족) vs 자살률 Pearson r: {pearson_r_neg:.4f} (p-value: {p_val_neg:.4e})\n\n")
    
    f.write("2. 성별에 따른 상관관계 (Pearson r)\n")
    for g in genders:
        f.write(f"  - 성별 '{g}':\n")
        f.write(f"    * 만족도 평균 점수 vs 자살률: r = {gender_corrs[g]['pearson_r']:.4f} (p-val: {gender_corrs[g]['pearson_p']:.4e})\n")
        f.write(f"    * 스피어만 순위상관계수: rs = {gender_corrs[g]['spearman_r']:.4f}\n")
        
    f.write("\n3. 연도별 지역간 만족도 점수와 자살률 상관관계 (전국 제외, 17개 시도간 비교)\n")
    for y in years:
        if y in year_corrs:
            f.write(f"  - {y}년: 만족도 평균 점수 vs 자살률 r = {year_corrs[y]['pearson_r']:.4f} (p-val: {year_corrs[y]['p_value']:.4f})\n")
            f.write(f"         부정 비율 vs 자살률 r = {year_corrs[y]['r_neg']:.4f} (p-val: {year_corrs[y]['p_neg']:.4f})\n")

print("Analysis summary saved.")
