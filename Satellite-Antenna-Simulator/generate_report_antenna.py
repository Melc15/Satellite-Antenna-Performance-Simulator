
"""
Satellite Phased-Array Antenna Performance Simulator

Includes:
- Baseline antenna performance
- Frequency sweep analysis
- Beam-steering sweep analysis
- Radiation pattern visualizations
- Fault injection analysis
- Automated requirements verification

"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

from antenna_config import (
    FREQUENCY_HZ,
    WAVELENGTH,
    ELEMENTS_X,
    ELEMENTS_Y,
    TOTAL_ELEMENTS,
    SPACING_X_METRES,
    SPACING_Y_METRES,
    STEERING_THETA_DEG,
    STEERING_PHI_DEG,
    ELEMENT_PATTERN_EXPONENT,
)

from performance_analysis import calculate_performance

from fault_simulation import (
    create_fault_scenarios,
    calculate_faulty_pattern,
)

from verification import run_verification, verification_results


#Report output configuration

REPORT_NAME = "Satellite_Antenna_Verification_Report.pdf"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


#Table styling helper function

def apply_table_style(table, font_size=8):

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )

    return table


#Image helper function

def make_image(path, width=6.2 * inch):

    if not os.path.exists(path):

        return Paragraph(
            f"Plot unavailable: {path}",
            getSampleStyleSheet()["Normal"]
        )

    img = Image(path)

    ratio = img.imageHeight / img.imageWidth

    img.drawWidth = width
    img.drawHeight = width * ratio

    return img


#Plot grid helper function

def make_plot_grid(paths):

    images = [
        make_image(path, width=3.0 * inch)
        for path in paths
    ]

    while len(images) < 4:
        images.append("")

    grid = Table(
        [
            [images[0], images[1]],
            [images[2], images[3]],
        ],
        colWidths=[3.1 * inch, 3.1 * inch]
    )

    grid.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )

    return grid


#Sweep table helper function

def make_sweep_table(
    csv_path,
    columns,
    headings,
    col_widths,
    formats
):

    if not os.path.exists(csv_path):

        return Paragraph(
            f"<b>Results unavailable:</b> {csv_path}. "
            "Run the corresponding sweep program before regenerating "
            "the report.",
            getSampleStyleSheet()["Normal"]
        )

    dataframe = pd.read_csv(csv_path)

    missing_columns = [
        column for column in columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        return Paragraph(
            f"<b>Invalid sweep results:</b> {csv_path} is missing "
            f"the expected columns: {', '.join(missing_columns)}.",
            getSampleStyleSheet()["Normal"]
        )

    rows = [headings]

    for _, record in dataframe.iterrows():

        row = []

        for column, number_format in zip(columns, formats):

            value = record[column]

            if pd.isna(value):
                formatted_value = "N/A"

            elif number_format == "angle":
                formatted_value = f"{value:.4f}"

            elif number_format == "three":
                formatted_value = f"{value:.3f}"

            elif number_format == "two":
                formatted_value = f"{value:.2f}"

            elif number_format == "one":
                formatted_value = f"{value:.1f}"

            else:
                formatted_value = str(value)

            row.append(formatted_value)

        rows.append(row)

    table = Table(
        rows,
        colWidths=col_widths,
        repeatRows=1
    )

    return apply_table_style(table, font_size=7)


#Fault results helper function

def get_fault_results():

    failed_elements, phase_errors = create_fault_scenarios()

    theta_values = np.linspace(0, 90, 721)

    ideal_pattern = np.asarray(
        calculate_faulty_pattern(theta_values)
    )

    failed_pattern = np.asarray(
        calculate_faulty_pattern(
            theta_values,
            failed_elements=failed_elements
        )
    )

    phase_pattern = np.asarray(
        calculate_faulty_pattern(
            theta_values,
            phase_errors=phase_errors
        )
    )

    combined_pattern = np.asarray(
        calculate_faulty_pattern(
            theta_values,
            failed_elements=failed_elements,
            phase_errors=phase_errors
        )
    )

    scenarios = {
        "Ideal antenna": ideal_pattern,
        "Failed elements": failed_pattern,
        "Phase errors": phase_pattern,
        "Combined faults": combined_pattern,
    }

    ideal_peak = np.max(ideal_pattern)

    results = {}

    for name, pattern in scenarios.items():

        peak = np.max(pattern)

        degradation = 10 * np.log10(
            max(peak, 1e-12) / max(ideal_peak, 1e-12)
        )

        results[name] = degradation

    return results, failed_elements


#Fault comparison plot helper function

def create_fault_plot():

    failed_elements, phase_errors = create_fault_scenarios()

    theta = np.linspace(0, 90, 721)

    scenarios = {
        "Ideal": calculate_faulty_pattern(theta),

        "Failed elements": calculate_faulty_pattern(
            theta,
            failed_elements=failed_elements
        ),

        "Phase errors": calculate_faulty_pattern(
            theta,
            phase_errors=phase_errors
        ),

        "Combined faults": calculate_faulty_pattern(
            theta,
            failed_elements=failed_elements,
            phase_errors=phase_errors
        ),
    }

    plt.figure(figsize=(10, 6))

    for name, pattern in scenarios.items():

        pattern = np.asarray(pattern)

        power_db = 10 * np.log10(
            np.maximum(pattern, 1e-12)
        )

        plt.plot(theta, power_db, label=name)

    plt.xlabel("Theta Angle (degrees)")
    plt.ylabel("Normalized Power (dB)")
    plt.title("Antenna Fault Simulation Comparison")

    plt.ylim(-40, 2)
    plt.xlim(0, 90)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "report_fault_comparison.png"
    )

    plt.savefig(path, dpi=200)
    plt.close()

    return path


#MAIN report generation function

def generate_report():

    print("\nGenerating satellite antenna verification report...")

    # Run automated verification
    run_verification()

    # Calculate baseline antenna performance
    performance = calculate_performance()

    # Calculate fault results
    fault_results, failed_elements = get_fault_results()

    # Generate fault comparison plot
    fault_plot = create_fault_plot()

    # Set up PDF document

    output_path = os.path.abspath(REPORT_NAME)

    document = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            spaceAfter=15,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=14,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
        )
    )

    story = []

    # Title page

    story.append(Spacer(1, 0.8 * inch))

    story.append(
        Paragraph(
            "Satellite Phased-Array<br/>"
            "Antenna Performance Simulator",
            styles["ReportTitle"]
        )
    )

    story.append(
        Paragraph(
            "Automated Engineering Verification Report",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 0.4 * inch))

    story.append(
        Paragraph(
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 0.2 * inch))

    story.append(
        Paragraph(
            "Test environment: Python-based theoretical simulation",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 0.3 * inch))

    story.append(
        Paragraph(
            "<b>Important:</b> This report presents results from an "
            "educational mathematical antenna simulation. No physical "
            "antenna, RF instrument, or satellite hardware was tested. ",
            styles["Normal"]
        )
    )

    story.append(PageBreak())

    # 1. Project Overview

    story.append(
        Paragraph("1. Project Overview", styles["SectionHeading"])
    )

    story.append(
        Paragraph(
            "This project implements a theoretical planar phased-array "
            "antenna model. The simulator calculates element positions, "
            "steering phases, radiation patterns, and selected antenna "
            "performance metrics. It also evaluates simulated component "
            "failures and phase errors through fault-injection scenarios. "
            "Frequency and beam-steering sweeps are used to examine how "
            "selected performance metrics change as operating conditions vary.",
            styles["Normal"]
        )
    )

    # 2. Antenna Configuration

    story.append(
        Paragraph("2. Antenna Configuration", styles["SectionHeading"])
    )

    configuration = [
        ["Parameter", "Configured Value"],
        ["Operating frequency", f"{FREQUENCY_HZ / 1e9:.2f} GHz"],
        ["Wavelength", f"{WAVELENGTH * 1000:.3f} mm"],
        ["Array dimensions", f"{ELEMENTS_X} x {ELEMENTS_Y}"],
        ["Total elements", str(TOTAL_ELEMENTS)],
        ["Element spacing X", f"{SPACING_X_METRES * 1000:.3f} mm"],
        ["Element spacing Y", f"{SPACING_Y_METRES * 1000:.3f} mm"],
        ["Commanded theta", f"{STEERING_THETA_DEG:.2f} degrees"],
        ["Commanded phi", f"{STEERING_PHI_DEG:.2f} degrees"],
        ["Element pattern exponent", str(ELEMENT_PATTERN_EXPONENT)],
    ]

    config_table = Table(
        configuration,
        colWidths=[3.0 * inch, 3.0 * inch]
    )

    story.append(apply_table_style(config_table))

    # 3. Baseline Antenna Performance

    story.append(
        Paragraph("3. Baseline Antenna Performance", styles["SectionHeading"])
    )

    performance_rows = [
        ["Performance Metric", "Simulated Result"],
        [
            "Commanded beam direction",
            f"{performance['commanded_theta_deg']:.3f} degrees"
        ],
        [
            "Peak beam direction",
            f"{performance['peak_theta_deg']:.3f} degrees"
        ],
        [
            "Pointing error",
            f"{performance['pointing_error_deg']:.3f} degrees"
        ],
        [
            "Half-power beamwidth",
            f"{performance['hpbw_deg']:.3f} degrees"
        ],
        [
            "Peak sidelobe level",
            (
                f"{performance['peak_sll_db']:.2f} dB"
                if performance['peak_sll_db'] is not None
                else "Not detected"
            )
        ],
        [
            "Left -3 dB crossing",
            f"{performance['left_half_power_deg']:.3f} degrees"
        ],
        [
            "Right -3 dB crossing",
            f"{performance['right_half_power_deg']:.3f} degrees"
        ],
    ]

    performance_table = Table(
        performance_rows,
        colWidths=[3.0 * inch, 3.0 * inch]
    )

    story.append(apply_table_style(performance_table))

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "Measurements are calculated from the simulated radiation "
            "pattern. Pointing error and sidelobe results are specific "
            "to the selected angular sampling and the phi = 0 degree cut.",
            styles["SmallText"]
        )
    )

    # 4. Frequency Sweep Analysis

    story.append(PageBreak())

    story.append(
        Paragraph("4. Frequency Sweep Analysis", styles["SectionHeading"])
    )

    story.append(
        Paragraph(
            "The frequency sweep evaluates the simulated antenna response "
            "at five operating frequencies between 18 and 22 GHz. The "
            "physical element positions and spacing remain fixed at the "
            "reference design values. Steering phases are recalculated "
            "at each frequency to maintain the commanded beam direction.",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 10))

    frequency_table = make_sweep_table(
        "frequency_sweep_results.csv",
        [
            "frequency_ghz",
            "peak_theta_deg",
            "pointing_error_deg",
            "hpbw_deg",
            "peak_sll_db",
        ],
        [
            "Frequency<br/>(GHz)",
            "Peak angle<br/>(deg)",
            "Pointing error<br/>(deg)",
            "HPBW<br/>(deg)",
            "Peak SLL<br/>(dB)",
        ],
        [
            1.0 * inch,
            1.15 * inch,
            1.25 * inch,
            1.0 * inch,
            1.0 * inch,
        ],
        ["two", "three", "three", "three", "two"]
    )

    story.append(frequency_table)

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Frequency sweep results and performance plots:",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 6))

    story.append(
        make_plot_grid([
            "frequency_pointing_error.png",
            "frequency_beamwidth.png",
            "frequency_sidelobe_level.png",
        ])
    )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "This is a phase-adjusted frequency sweep, not a fixed-phase "
            "beam-squint experiment. It does not establish measured "
            "frequency-dependent antenna performance.",
            styles["SmallText"]
        )
    )

    # 5. Beam-Steering Sweep Analysis

    story.append(PageBreak())

    story.append(
        Paragraph("5. Beam-Steering Sweep Analysis", styles["SectionHeading"])
    )

    story.append(
        Paragraph(
            "The beam-steering sweep evaluates the antenna at commanded "
            "steering angles from 0 to 50 degrees, in 10-degree increments. "
            "Operating frequency and physical element spacing remain fixed. "
            "Parabolic interpolation is applied around the sampled main-beam "
            "maximum to estimate its peak direction between angular samples.",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 10))

    steering_table = make_sweep_table(
        "beam_steering_results.csv",
        [
            "commanded_angle_deg",
            "measured_peak_deg",
            "pointing_error_deg",
            "hpbw_deg",
            "peak_sll_db",
        ],
        [
            "Command<br/>(deg)",
            "Peak angle<br/>(deg)",
            "Pointing error<br/>(deg)",
            "HPBW<br/>(deg)",
            "Peak SLL<br/>(dB)",
        ],
        [
            1.0 * inch,
            1.15 * inch,
            1.25 * inch,
            1.0 * inch,
            1.0 * inch,
        ],
        ["one", "angle", "angle", "three", "two"]
    )

    story.append(steering_table)

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Beam-steering sweep results and performance plots:",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 6))

    story.append(
        make_plot_grid([
            "beam_steering_pointing_error.png",
            "beam_steering_beamwidth.png",
            "beam_steering_sidelobe_level.png",
        ])
    )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "The interpolated peak is a numerical estimate based on the "
            "simulated angular pattern. It is not a physical measurement "
            "or a guarantee of antenna pointing accuracy.",
            styles["SmallText"]
        )
    )

    # 6. Radiation Pattern Visualizations

    story.append(PageBreak())

    story.append(
        Paragraph("6. Radiation Pattern Visualizations", styles["SectionHeading"])
    )

    story.append(
        Paragraph(
            "The following figures illustrate the simulated antenna "
            "radiation characteristics for the baseline configuration.",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 10))

    story.append(make_image("radiation_pattern_2d.png"))

    story.append(Spacer(1, 12))

    story.append(make_image("radiation_pattern_3d.png"))

    # 7. Fault Injection Analysis

    story.append(PageBreak())

    story.append(
        Paragraph("7. Fault Injection Analysis", styles["SectionHeading"])
    )

    story.append(
        Paragraph(
            f"The failed-element scenario disabled {len(failed_elements)} "
            f"of {TOTAL_ELEMENTS} antenna elements "
            f"({100 * len(failed_elements) / TOTAL_ELEMENTS:.2f}%). "
            "Random phase errors were also introduced to evaluate their "
            "effects on simulated peak power. The phase-error scenario "
            "uses a reproducible random seed.",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 10))

    fault_rows = [
        ["Fault Scenario", "Relative Peak Power"]
    ]

    for name, value in fault_results.items():

        fault_rows.append([
            name,
            f"{value:.2f} dB"
        ])

    fault_table = Table(
        fault_rows,
        colWidths=[3.0 * inch, 3.0 * inch]
    )

    story.append(apply_table_style(fault_table))

    story.append(Spacer(1, 15))

    story.append(make_image(fault_plot))

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "Relative peak power is compared with the ideal simulated "
            "pattern. These scenarios represent software-injected faults, "
            "not faults measured on physical antenna hardware.",
            styles["SmallText"]
        )
    )

    # 8. Automated Requirements Verification

    story.append(PageBreak())

    story.append(
        Paragraph(
            "8. Automated Requirements Verification",
            styles["SectionHeading"]
        )
    )

    results = verification_results

    verification_rows = [
        ["Requirement", "Measured Result", "Criterion", "Status"]
    ]

    for result in results:

        verification_rows.append([
            str(result.get("requirement", "")),
            str(result.get("measured_value", "")),
            str(result.get("acceptance_criteria", "")),
            str(result.get("status", "")),
        ])

    verification_table = Table(
        verification_rows,
        colWidths=[
            1.1 * inch,
            1.5 * inch,
            2.2 * inch,
            1.0 * inch,
        ],
        repeatRows=1
    )

    story.append(apply_table_style(verification_table))

    # 9. Conclusion

    story.append(
        Paragraph("9. Verification Conclusion", styles["SectionHeading"])
    )

    passed = sum(
        1 for result in results
        if str(result.get("status", "")).upper() == "PASS"
    )

    total = len(results)

    story.append(
        Paragraph(
            f"The automated verification process completed with "
            f"{passed} of {total} project-defined acceptance checks passing.",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "<b>Limitations:</b> The simulator uses an idealized mathematical "
            "array model and an assumed element radiation pattern. It does "
            "not include full-wave electromagnetic analysis, hardware "
            "tolerances, measured antenna data, environmental testing, "
            "or validation against an actual satellite antenna. The "
            "frequency sweep recalculates steering phases at each frequency. "
            "Sidelobe analysis is based on the phi = 0 degree pattern cut. "
            "All acceptance criteria are chosen project requirements by Melissa Choloniuk. ",
            styles["SmallText"]
        )
    )

    # Build the PDF document

    document.build(story)

    print("\n" + "=" * 60)
    print("Verification report generation complete.")
    print("=" * 60)

    print(f"Report: {output_path}")
    print(f"Acceptance checks passed: {passed}/{total}")

    print("\nIncluded sweep data:")

    print(
        "Frequency sweep CSV:",
        "FOUND" if os.path.exists("frequency_sweep_results.csv") else "MISSING"
    )

    print(
        "Beam-steering sweep CSV:",
        "FOUND" if os.path.exists("beam_steering_results.csv") else "MISSING"
    )

    print("=" * 60)


if __name__ == "__main__":
    generate_report()
