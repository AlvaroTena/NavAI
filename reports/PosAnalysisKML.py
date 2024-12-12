import argparse
import os
import subprocess


def execute_pos_analysis(base_dir, scenario, ai_type, rtppp_file):
    pos_file = f"{base_dir}/{scenario}/{ai_type}/{rtppp_file}"
    ref_file = f"data/scenarios/{scenario}/INPUTS/references/kinematic_reference.txt"
    output_dir = f"{base_dir}/{scenario}/{ai_type}/PosAnalysis"

    command = [
        "python3",
        "../subprojects/common_lib/test_scripts/MonitoringTools/PosFileAnalysis.py",
        "-pos_type",
        "RTPPP_GSHARP",
        "-pos_file",
        pos_file,
        "-ref_mode",
        "kinematic",
        "-ref_type",
        "SPAN_FILE",
        "-ref_file",
        ref_file,
        "-o",
        output_dir,
        "-info_in_plots_legend",
    ]

    print(f"Executing command: {' '.join(command)}")
    subprocess.run(command, check=True)


def execute_kipos2kml(base_dir, scenario, ai_type, rtppp_file):
    input_file = f"{base_dir}/{scenario}/{ai_type}/{rtppp_file}"
    output_file = input_file.replace(".txt", ".kml")
    final_param = "1" if ai_type == "AI" else "2"

    command = ["python3", "reports/KIPOS2KML.py", input_file, output_file, final_param]

    print(f"Executing command: {' '.join(command)}")
    subprocess.run(command, check=True)


def find_scenarios(base_dir):
    scenarios = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name.startswith("scenario"):
                scenarios.append(dir_name)
    return sorted(scenarios)


def main(base_dir):
    scenarios = find_scenarios(base_dir)

    for scenario in scenarios:
        for ai_type in ["AI", "noAI"]:
            scenario_dir = os.path.join(base_dir, scenario, ai_type)
            if os.path.isdir(scenario_dir):
                for file_name in os.listdir(scenario_dir):
                    if file_name.startswith("rtppp") and file_name.endswith(".txt"):
                        try:
                            # Ejecutar el análisis de posición
                            execute_pos_analysis(base_dir, scenario, ai_type, file_name)
                            # Ejecutar la conversión a KML
                            execute_kipos2kml(base_dir, scenario, ai_type, file_name)
                        except subprocess.CalledProcessError as e:
                            print(
                                f"An error occurred while executing the scripts for {scenario} with {ai_type}."
                            )
                            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process scenarios and files.")
    parser.add_argument(
        "base_dir", type=str, help="The base directory containing scenarios."
    )
    args = parser.parse_args()

    main(args.base_dir)
