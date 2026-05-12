# Fine-Tuning Results Report

## Summary Statistics

| Dataset                        |   Pretrained MSE |   Pretrained MAE |   Scratch MSE |   Scratch MAE |   MSE Improvement % |   MAE Improvement % |
|:-------------------------------|-----------------:|-----------------:|--------------:|--------------:|--------------------:|--------------------:|
| Metro                          |         1.03586  |         0.883736 |      1.06796  |      0.884995 |             3.00522 |            0.142163 |
| Beijing_PM25                   |         0.833965 |         0.677759 |      0.714738 |      0.64492  |           -16.6813  |           -5.09196  |
| Environmental_Sensor_Telemetry |         1.81808  |         1.06218  |      1.9282   |      1.09864  |             5.71124 |            3.31933  |

## Key Findings

- **Average MSE Improvement**: -2.65%
- **Average MAE Improvement**: -0.54%

## Dataset-Specific Results

### Metro

**Pretrained Model:**
- Test MSE: 1.035863
- Test MAE: 0.883736

**From-Scratch Baseline:**
- Test MSE: 1.067957
- Test MAE: 0.884995

**Improvement:**
- MSE: +3.01%
- MAE: +0.14%

### Beijing_PM25

**Pretrained Model:**
- Test MSE: 0.833965
- Test MAE: 0.677759

**From-Scratch Baseline:**
- Test MSE: 0.714738
- Test MAE: 0.644920

**Improvement:**
- MSE: -16.68%
- MAE: -5.09%

### Environmental_Sensor_Telemetry

**Pretrained Model:**
- Test MSE: 1.818076
- Test MAE: 1.062176

**From-Scratch Baseline:**
- Test MSE: 1.928200
- Test MAE: 1.098644

**Improvement:**
- MSE: +5.71%
- MAE: +3.32%

## Methodology

1. **Linear Probe**: Frozen pre-trained encoder, train only forecasting head
2. **Full Fine-tuning**: Unfreeze all layers with lower learning rate (5e-5)
3. **Baseline**: Identical architecture trained from scratch
4. **Evaluation**: Test MSE and MAE on 20% holdout test set

## Conclusion

Pre-trained models show mixed results, with an average **2.65% degradation** vs from-scratch models.
This may indicate that the pre-training task (masked reconstruction) is not well-aligned with the downstream forecasting task.
