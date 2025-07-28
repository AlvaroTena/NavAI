import argparse
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

gps_start_date = pd.to_datetime("1980 1 6 0 0 0.000000", format="%Y %m %d %H %M %S.%f")


def load_reward_file(filepath):
    # Leer el CSV con la cabecera correcta
    df = pd.read_csv(filepath)

    # Convertir la columna epoch a datetime
    df["time"] = pd.to_datetime(
        df["epoch"], format="%Y   %m  %d   %H  %M   %S.%f", errors="coerce"
    )

    # Extraer los componentes del cumulative_reward (que viene como string de array)
    df["cumulative_reward"] = df["cumulative_reward"].apply(
        lambda x: np.array([float(n) for n in x.strip("[]").split()])
    )
    df["north"] = df["cumulative_reward"].apply(lambda x: x[0])
    df["east"] = df["cumulative_reward"].apply(lambda x: x[1])
    df["up"] = df["cumulative_reward"].apply(lambda x: x[2])

    # Filtrar filas inválidas
    df = df[(df["time"] != gps_start_date) & df["time"].notna()]
    df.sort_values("time", inplace=True)  # Ensure the data is sorted by time
    return df


def load_restart_times(scenario_dir):
    """
    Load restart times from state_transition_writer_*.txt file.
    Look for lines with the INIT_MON signal to determine when to add a vertical line.
    """
    restart_times = []
    # Find the relevant state_transition_writer_*.txt file
    for dirpath, _, filenames in os.walk(scenario_dir):
        for filename in filenames:
            if filename.startswith("state_transition_writer") and filename.endswith(
                ".txt"
            ):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, "r") as file:
                    for line in file:
                        if "INIT_MON" in line:
                            # Extract the timestamp from the line
                            parts = line.split()
                            if len(parts) >= 2:
                                timestamp_str = parts[0] + " " + parts[1]
                                try:
                                    # month/day/year
                                    timestamp = pd.to_datetime(
                                        timestamp_str, format="%m/%d/%y %H:%M:%S.%f"
                                    )
                                except ValueError as e:
                                    print(
                                        f"Could not parse timestamp from line: {line} - {e}"
                                    )
                                    continue
                                restart_times.append(timestamp)
    return restart_times


def add_vertical_lines(fig, restart_times, min_time, max_time):
    """
    Add vertical lines based on restart times, ensuring they are within the time range of the rewards data.
    """
    for time_point in restart_times:
        # Check if the restart time is within the range of the reward times
        if min_time <= time_point <= max_time:
            fig.add_shape(
                go.layout.Shape(
                    type="line",
                    x0=time_point,
                    y0=0,
                    x1=time_point,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(color="Red", dash="dash"),
                )
            )
            print(f"Added reconvergence line at {time_point}")


def create_and_save_plot_for_reward_file(reward_file, scenario_dir, output_path):
    df = load_reward_file(reward_file)

    if df.empty:
        print(f"No data to plot in file: {reward_file}")
        return

    fig = go.Figure()

    # Añadir una traza para cada componente
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["north"],
            mode="lines",
            name="North",
            line=dict(color="red"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["east"],
            mode="lines",
            name="East",
            line=dict(color="green"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["up"],
            mode="lines",
            name="Up",
            line=dict(color="blue"),
        )
    )

    # Load the restart times and add vertical lines for them
    restart_times = load_restart_times(scenario_dir)
    add_vertical_lines(fig, restart_times, df["time"].min(), df["time"].max())

    fig.update_layout(
        title_text=os.path.basename(reward_file),
        xaxis_title="Time",
        yaxis_title="Reward Components (NEU)",
        showlegend=True,
    )

    # Save the plot as a PNG file with the same name as the CSV file
    fig.write_image(output_path, width=1600, height=900)
    print(f"Saved plot to {output_path}")


def find_reward_files(scenario_dir):
    reward_files = []
    for dirpath, _, filenames in os.walk(scenario_dir):
        for filename in filenames:
            if filename.startswith("reward") and filename.endswith(".csv"):
                reward_files.append(os.path.join(dirpath, filename))
    return reward_files


def find_scenarios(base_dir):
    scenarios = []
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name.startswith("scenario"):
                scenarios.append(os.path.join(root, dir_name))
    return sorted(scenarios)


def process_scenarios(base_dir):
    scenarios = find_scenarios(base_dir)
    for scenario in scenarios:
        reward_files = find_reward_files(scenario)
        if reward_files:
            print(f"Processing scenario: {os.path.basename(scenario)}")
            for reward_file in reward_files:
                output_path = os.path.join(
                    os.path.dirname(reward_file),
                    f"{os.path.splitext(os.path.basename(reward_file))[0]}.png",
                )
                create_and_save_plot_for_reward_file(reward_file, scenario, output_path)
        else:
            print(f"No reward files found in scenario: {os.path.basename(scenario)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process scenarios and files.")
    parser.add_argument(
        "base_dir", type=str, help="The base directory containing scenarios."
    )
    args = parser.parse_args()

    process_scenarios(args.base_dir)
