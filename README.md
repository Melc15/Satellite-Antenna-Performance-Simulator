# Satellite-Antenna-Performance-Simulator
# Satellite Phased-Array Antenna Performance Simulator

This Python-based program models a 16 × 16 planar phased-array antenna operating at a representative frequency of 20 GHz. It uses array-factor calculations and an assumed element radiation pattern to investigate beam steering, radiation characteristics, and simulated antenna faults.

The project demonstrates the use of Python for antenna performance analysis, parameter sweeps, fault injection, automated verification, and technical reporting.

## Key Features

- **Phased-array modelling:** Calculates element positions, steering phases, and array radiation patterns.
- **Radiation pattern analysis:** Estimates main-beam direction, pointing error, half-power beamwidth (HPBW), and peak sidelobe level (SLL).
- **Frequency sweep:** Evaluates simulated performance from 18 to 22 GHz while maintaining fixed physical element spacing and recalculating steering phases at each frequency.
- **Beam-steering sweep:** Evaluates commanded steering angles from 0° to 50° and uses parabolic interpolation to estimate the main-beam peak.
- **Fault injection:** Simulates failed antenna elements, random phase errors, and combined fault conditions.
- **Automated verification:** Evaluates six project-defined acceptance criteria.
- **Requirements traceability:** Maps verification criteria to implementation and test methods.
- **Technical reporting:** Generates a PDF report containing configuration, simulation results, plots, and verification outcomes.

## Antenna Configuration

| Parameter | Configuration |
|---|---:|
| Array geometry | Planar |
| Array dimensions | 16 × 16 |
| Total elements | 256 |
| Reference operating frequency | 20 GHz |
| Element spacing | 0.5λ |
| Reference steering angle | 30° |
| Element pattern | Cosine-power model |

## Technologies

- Python
- NumPy
- SciPy
- Matplotlib
- Pandas
- Pytest
- ReportLab

## Verification

The simulator includes six illustrative acceptance checks covering:

| Requirement | Verification objective |
|---|---|
| ANT-001 | Beam pointing error |
| ANT-002 | Half-power beamwidth |
| ANT-003 | Peak sidelobe level |
| ANT-004 | Failed-element degradation |
| ANT-005 | Phase-error degradation |
| ANT-006 | Combined fault degradation |

The current implementation has passed 12 automated unit tests and all six project-defined simulated acceptance checks.


## Simulation Limitations

- Uses an idealized mathematical array-factor model.
- Uses an assumed cosine-power element radiation pattern.
- Does not perform full-wave electromagnetic simulation.
- Does not include measured antenna or RF instrument data.
- Does not model all hardware, thermal, environmental, or manufacturing effects.
- Sidelobe analysis is based on a φ = 0° radiation-pattern cut.
- Frequency-sweep steering phases are recalculated at each frequency; the sweep is not a fixed-phase beam-squint analysis.
- Results are not validated against MDA AURORA or IRIS² hardware.

## Project Objective

The objective is to demonstrate a software-based engineering workflow for satellite antenna analysis and verification, combining mathematical modelling, performance characterization, fault simulation, automated testing, and requirements traceability.
