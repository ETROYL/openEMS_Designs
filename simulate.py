import os
import argparse
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from design import build_design
from design import load_config

def main():
    parser = argparse.ArgumentParser(description="Generic openEMS Simulation Runner")
    parser.add_argument("--config", type=str, default="./bowtie.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()
    config_file = getattr(args, 'config', 'bowtie.yaml')
    cfg = load_config(config_file)
    dir = config_file[0:-5] # str(cfg.get('simulation', {}).get('path', 'tmp'))
    fstart = (cfg.get('simulation', {}).get('fstart', 0.01))*1e9
    fstop = (cfg.get('simulation', {}).get('fstop', 1))*1e9


    f0 = (fstart + fstop) / 2.0
    fc = (fstop - fstart)

    CSX = ContinuousStructure()
    FDTD = openEMS(EndCriteria=1e-4)
    FDTD.SetCSX(CSX)
    FDTD.SetGaussExcite(f0, fc)

    # Set Boundary Conditions (PML_8)
    FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])

    # FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'MUR'])

    ports = build_design(CSX, FDTD, config_file)

    if not os.path.exists(dir):
        os.makedirs(dir)

    xml_file = os.path.join(dir, 'bowtie.xml')
    CSX.Write2XML(xml_file)
    print(f"CSXCAD geometry saved to: {xml_file}")

    print("Running openEMS solver...")
    FDTD.Run(dir)
    print("Simulation finished. Data written to directory:", dir)

if __name__ == "__main__":
    main()