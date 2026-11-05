
"""
Automated Verification Report Generator
This script generates a PDF report summarizing the results of the satellite phased-array antenna simulation, including performance metrics, radiation pattern visualizations, fault injection analysis, and automated requirements verification.
"""

import os
from datetime import datetime

import numpy as np
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
    SPEED_OF_LIGHT,
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

# Report configuration

REPORT_NAME = "Satellite_Antenna_Verification_Report.pdf"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


#Image helper function for report generation

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


#Fault results calculation

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


#Fault comparison plot generation

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


#Main report generation function

def generate_report():

    print("\nGenerating satellite antenna verification report...")

    # Run automated verification
    run_verification()

    # Calculate antenna performance
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
            "Automated Verification Report",
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

    # Project overview

    story.append(
        Paragraph("1. Project Overview", styles["SectionHeading"])
    )

    story.append(
        Paragraph(
            "This project implements a theoretical planar phased-array "
            "antenna model. The simulator calculates element positions, "
            "steering phases, radiation patterns, and selected antenna "
            "performance metrics. It also evaluates simulated component "
            "failures and phase errors through fault-injection scenarios.",
            styles["Normal"]
        )
    )

    # Antenna configuration

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

    config_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("PADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.append(config_table)

    # Performance analysis

    story.append(
        Paragraph("3. Antenna Performance Analysis", styles["SectionHeading"])
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

    performance_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.append(performance_table)

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "Measurements are calculated from the simulated radiation "
            "pattern. The pointing error and sidelobe results are specific "
            "to the selected angular sampling and the phi = 0 degree cut.",
            styles["SmallText"]
        )
    )

    # Radiation pattern plots

    story.append(
        Paragraph("4. Radiation Pattern Visualizations", styles["SectionHeading"])
    )

    story.append(
        Paragraph(
            "The figures below illustrate the simulated antenna "
            "radiation characteristics.",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 10))

    story.append(make_image("radiation_pattern_2d.png"))

    story.append(Spacer(1, 12))

    story.append(make_image("radiation_pattern_3d.png"))

    # Fault injection analysis

    story.append(PageBreak())

    story.append(
        Paragraph("5. Fault Injection Analysis", styles["SectionHeading"])
    )

    story.append(
        Paragraph(
            f"The failed-element scenario disabled {len(failed_elements)} "
            f"of {TOTAL_ELEMENTS} antenna elements. Random phase errors "
            "were also introduced to evaluate their effects on simulated "
            "peak power.",
            styles["Normal"]
        )
    )

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

    fault_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.append(fault_table)

    story.append(Spacer(1, 15))

    story.append(make_image(fault_plot))

    # Verification results

    story.append(PageBreak())

    story.append(
        Paragraph("6. Automated Requirements Verification", styles["SectionHeading"])
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

    verification_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )

    story.append(verification_table)

    # Conclusion

    story.append(
        Paragraph("7. Verification Conclusion", styles["SectionHeading"])
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
            "tolerances, measured antenna data, environmental testing, or "
            "validation against an actual satellite antenna. Acceptance "
            "criteria were chosen by Melissa Choloniuk for educational purposes and may not reflect real-world satellite antenna requirements.",
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
    print("=" * 60)


if __name__ == "__main__":
    generate_report()
