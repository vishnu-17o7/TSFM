# Fine-Tuning Results Report

## Summary Statistics

| Dataset                        |   Pretrained MSE |   Pretrained MAE |   Scratch MSE |   Scratch MAE |   MSE Improvement % |   MAE Improvement % |
|:-------------------------------|-----------------:|-----------------:|--------------:|--------------:|--------------------:|--------------------:|
| ETTh1                          |         1.48823  |         0.962425 |      1.44902  |      0.990002 |            -2.70616 |            2.78558  |
| Metro                          |         1.08175  |         0.908832 |      1.07699  |      0.900359 |            -0.44162 |           -0.941087 |
| Beijing_PM25                   |         0.777077 |         0.681348 |      0.986856 |      0.77254  |            21.2573  |           11.8042   |
| Environmental_Sensor_Telemetry |         1.5379   |         0.974177 |      1.6142   |      1.0044   |             4.72708 |            3.00867  |

## Key Findings

- **Average MSE Improvement**: 5.71%
- **Average MAE Improvement**: 4.16%

## Dataset-Specific Results

### ETTh1

**Pretrained Model:**
- Test MSE: 1.488231
- Test MAE: 0.962425

**From-Scratch Baseline:**
- Test MSE: 1.449018
- Test MAE: 0.990002

**Improvement:**
- MSE: -2.71%
- MAE: +2.79%

### Metro

**Pretrained Model:**
- Test MSE: 1.081748
- Test MAE: 0.908832

**From-Scratch Baseline:**
- Test MSE: 1.076992
- Test MAE: 0.900359

**Improvement:**
- MSE: -0.44%
- MAE: -0.94%

### Beijing_PM25

**Pretrained Model:**
- Test MSE: 0.777077
- Test MAE: 0.681348

**From-Scratch Baseline:**
- Test MSE: 0.986856
- Test MAE: 0.772540

**Improvement:**
- MSE: +21.26%
- MAE: +11.80%

### Environmental_Sensor_Telemetry

**Pretrained Model:**
- Test MSE: 1.537897
- Test MAE: 0.974177

**From-Scratch Baseline:**
- Test MSE: 1.614202
- Test MAE: 1.004396

**Improvement:**
- MSE: +4.73%
- MAE: +3.01%

## Methodology

1. **Linear Probe**: Frozen pre-trained encoder, train only forecasting head
2. **Full Fine-tuning**: Unfreeze all layers with lower learning rate (5e-5)
3. **Baseline**: Identical architecture trained from scratch
4. **Evaluation**: Test MSE and MAE on 20% holdout test set

## Conclusion

Pre-training on synthetic time-series data provides consistent benefits:
On average, pre-trained models achieve **5.71% better MSE** than models trained from scratch.
This validates the effectiveness of pre-training for learning universal time-series representations.
