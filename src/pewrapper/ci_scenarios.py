import glob
import os
import shutil
import subprocess
from multiprocessing import Pool

scen_path = os.path.join(os.getcwd(), "data/scenarios/")
scenarios = sorted(
    [
        os.path.join(scen_path, scen)
        for scen in os.listdir(scen_path)
        if scen.startswith("scenario_") and os.path.isdir(os.path.join(scen_path, scen))
    ]
)

sessions_path = [os.path.join(scen, "CONFIG/session.ini") for scen in scenarios]
outputs_path = [os.path.join(scen, "out/") for scen in scenarios]


def process_scenario_wrapper(args):
    scen, session, output = args

    print(f"Launched {scen}...", flush=True)
    cmd = f"python3 src/pewrapper/pe_wrapper.py -s {session} -o {output} -g ERROR"
    result = subprocess.run(cmd, shell=True, cwd=os.getcwd())

    if result.returncode == 0:
        target_dir = os.path.join(
            scen,
            "INPUTS/AI",
        )
        os.makedirs(target_dir, exist_ok=True)
        for file in glob.glob(f"{output}Tracing_Output/AI_Multipath_*.txt"):
            shutil.copy(
                file,
                target_dir,
            )
        shutil.rmtree(output)
        print(f"Finished {scen}", flush=True)
    else:
        print(f"Error processing {scen}", flush=True)

    return scen, result.returncode


with Pool(15) as pool:
    results = pool.map(
        process_scenario_wrapper,
        zip(scenarios, sessions_path, outputs_path),
    )

scen_fail = [scen for scen, result in results if result != 0]
print("Failed scenarios:", scen_fail)
