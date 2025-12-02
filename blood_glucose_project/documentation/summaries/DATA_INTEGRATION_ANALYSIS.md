# Data Integration Analysis: Critical SEQN Mismatch Issue

**Date:** October 27, 2025  
**Issue:** Zero overlap between glucose targets and lifestyle data  
**Root Cause:** NHANES survey cycle mismatch  

## Critical Finding: SEQN Range Mismatch

### Data Source SEQN Ranges:
- **Glucose/HbA1c Targets:** 109264 - 124822 (NHANES 2017-2020)
- **Physical Activity Data:** 62161 - 83731 (NHANES 2011-2014)  
- **Dietary Data:** 62161 - 83731 (NHANES 2011-2014)
- **Demographics:** 109263 - 124822 (NHANES 2017-2020)

**Result:** ZERO participants overlap between glucose data and lifestyle data

## Data Integration Status

### Successful Merges:
- **Glucose + Demographics:** 4,732 participants (100% overlap)
- **Activity + Dietary:** 14,693 participants (same NHANES cycles)

### Failed Merges:
- **Glucose + Activity:** 0 participants (different NHANES cycles)
- **Glucose + Dietary:** 0 participants (different NHANES cycles)

## Feature Variance Analysis Results

### Features with Variance (4 total):
1. **Age** - Demographics (2017-2020)
2. **Gender** - Demographics (2017-2020)  
3. **Race/Ethnicity** - Demographics (2017-2020)
4. **Education Level** - Demographics (2017-2020)

### Features with Zero Variance (15 total):
**Physical Activity Features (12):**
- Total activity counts, wear time, MVPA, sedentary time, etc.
- **Reason:** No participants overlap with glucose data

**Dietary Features (1):**
- Daily vitamin D intake
- **Reason:** No participants overlap with glucose data

**Interaction Features (2):**
- Age × activity, gender × sedentary interactions
- **Reason:** Activity data missing due to SEQN mismatch

## Solutions to Fix Data Integration

### Option 1: Use Matching NHANES Cycles (RECOMMENDED)
**Load glucose data from 2011-2014 to match activity/dietary data:**
- Use `2011-2012_GLU_G.csv` and `2011-2012_GHB_G.csv`
- Use `2013-2014_GLU_H.csv` and `2013-2014_GHB_H.csv`
- **Expected Result:** ~14,000 participants with complete lifestyle + glucose data

### Option 2: Load Activity/Dietary from 2017-2020
**Find and load 2017-2020 activity/dietary files:**
- Look for `P_PAX*.xpt` (physical activity) files
- Look for `P_DR1TOT.xpt`, `P_DR2TOT.xpt` (dietary) files
- **Challenge:** May not be available in current data directory

### Option 3: Multi-Cycle Analysis
**Combine multiple NHANES cycles:**
- Use 2011-2014 data for lifestyle model validation
- Use 2017-2020 data for demographics-only model
- Compare results across cycles

## Immediate Action Plan

### Step 1: Load Matching Glucose Data (2011-2014)
```python
# Load glucose data from matching cycles
glucose_2011_2012 = pd.read_csv('processed_data_new/2011-2012_GLU_G.csv')
glucose_2013_2014 = pd.read_csv('processed_data_new/2013-2014_GLU_H.csv')
hba1c_2011_2012 = pd.read_csv('processed_data_new/2011-2012_GHB_G.csv') 
hba1c_2013_2014 = pd.read_csv('processed_data_new/2013-2014_GHB_H.csv')
```

### Step 2: Verify SEQN Overlap
```python
# Check SEQN ranges match activity data (62161 - 83731)
glucose_seqn_range = combined_glucose['SEQN'].min(), combined_glucose['SEQN'].max()
activity_seqn_range = activity_data['SEQN'].min(), activity_data['SEQN'].max()
```

### Step 3: Rebuild Complete Dataset
- Merge 2011-2014 glucose + activity + dietary + demographics
- Expected: ~10,000+ participants with all features
- All 27 engineered features should have variance

## Expected Results After Fix

### Complete Feature Set (27 features):
**Demographics (5):** Age, gender, race, education, BMI
**Physical Activity (12):** Total activity, MVPA, sedentary time, ratios, etc.
**Dietary (6):** Calories, carbs, fats, proteins, nutrients
**Interaction (4):** Age×activity, BMI×MVPA, etc.

### Expected Feature Importance Ranking:
1. **Age** (confirmed dominant)
2. **BMI** (expected #2 predictor)
3. **Total Physical Activity** (expected #3)
4. **Dietary Carbohydrates** (expected #4)
5. **Race/Ethnicity** (confirmed important)
6. **Education Level** (confirmed important)
7. **Gender** (confirmed modest effect)

## Clinical Impact

### Current Limitation:
- Model reduced to demographics-only prediction
- Missing key diabetes risk factors (BMI, activity, diet)
- Limited clinical utility for screening

### After Fix:
- Complete lifestyle-based screening model
- All major diabetes risk factors included
- Clinically meaningful feature importance analysis
- Deployable screening tool for population health

## Files Generated:
- `fixed_comprehensive_dataset.csv` - Current demographics-only dataset
- `fixed_feature_analysis.csv` - Feature variance analysis
- **Next:** `complete_lifestyle_dataset.csv` - After SEQN fix

## Conclusion

**Root Cause Identified:** NHANES survey cycle mismatch prevents data integration

**Solution:** Load glucose data from 2011-2014 to match existing activity/dietary data

**Expected Impact:** Transform from 4-feature demographics model to 27-feature comprehensive lifestyle model

**Timeline:** Can be fixed by Monday meeting with proper NHANES cycle alignment

---

**Status:** Critical data integration issue identified and solution planned
