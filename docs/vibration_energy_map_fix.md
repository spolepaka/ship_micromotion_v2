# Vibration Energy Map Generation Issue

## Problem Description

The ship micromotion analysis pipeline is not generating vibration energy maps, which are crucial for identifying and characterizing vibrating areas on ships. Analysis of the logs and code reveals several issues:

1. **Missing Calculation Step**: The code attempts to plot vibration energy maps but doesn't explicitly call the function to calculate them first. This results in the error message: "No vibration energy map available to plot".

2. **Demo Data Path Issues**: When using synthetic demo data, the pipeline fails earlier with the error "No displacement maps available", preventing any further processing including vibration energy map generation.

3. **Incomplete Processing Flow**: The pipeline correctly processes displacement maps and time series analysis, but doesn't integrate vibration energy calculation as a required step.

## Impact

Without vibration energy maps, users cannot:
- Visualize areas of high mechanical activity on the ship
- Detect mechanical signatures that could identify ship type/class
- Distinguish between different vibration sources (engines, generators, etc.)
- Perform comprehensive micromotion analysis

## Proposed Solution

The fix will consist of the following changes:

1. **Add Explicit Vibration Energy Calculation**: Implement a call to `estimator.calculate_vibration_energy_map()` before attempting to plot it.

2. **Fix Demo Data Path**: Ensure that displacement maps are properly generated for synthetic demo data by adding the missing displacement estimation step.

3. **Improve Error Handling**: Modify the code to continue processing even if certain steps fail, allowing partial results to be generated and displayed.

4. **Process Flow Enhancement**: Restructure the workflow to include vibration energy calculation as a standard step in the processing pipeline.

## Implementation Details

The fix will involve modifying:
- `app.py`: To add the missing vibration energy calculation step and improve error handling
- `micromotion_estimator.py`: To ensure the `calculate_vibration_energy_map` method works correctly
- Demo data generation: To properly include displacement estimation in the synthetic data path

By implementing these changes, the pipeline will generate comprehensive vibration energy maps for both real and synthetic data, completing the full analysis workflow as designed. 