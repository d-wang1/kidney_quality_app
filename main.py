import typer
import json
from files import get_kpmp_files, get_cancer_files, get_wustl_files, get_wustl_dir, get_kpmp_dir, get_cancer_dir, load_csv
from pipelines import processAndSave, run_all_files
from methods import writeToProcess, readProcess, clearProcess

#TODO: Auto-create process.txt file
#TODO: Remove need for consistent dir in file structure
#TODO: README.md

def load_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

app = typer.Typer()


@app.command()
def test_load_files(config: str = typer.Option("config.json", help="The JSON configuration file")):
    config = load_config(config)
    match config["source"]:
        case "kpmp":
            files = get_kpmp_files()
            dir = get_kpmp_dir()
        case "cancer":
            files = get_cancer_files()
            dir = get_cancer_dir()
        case "gl":
            files, _ = get_wustl_files()
            dir = get_wustl_dir()
        case "mts":
            _, files = get_wustl_files()
            dir = get_wustl_dir()
        case "csv":
            files = load_csv(config["file"])
            dir = config["dir"]
        case _:
            raise ValueError(f"Invalid source: {config['source']}")
    print(files)


@app.command()
def run(config: str = typer.Option("config.json", help="The JSON configuration file")):
    config = load_config(config)
    start_from_beginning = False
    match config["source"]:
        case "kpmp":
            files = get_kpmp_files()
            dir = get_kpmp_dir()
        case "cancer":
            files = get_cancer_files()
            dir = get_cancer_dir()
        case "gl":
            files, _ = get_wustl_files()
            dir = get_wustl_dir()
        case "mts":
            _, files = get_wustl_files()
            dir = get_wustl_dir()
        case "csv":
            files = load_csv(config["file"])
            dir = config["dir"]
            start_from_beginning = True
        case _:
            raise ValueError(f"Invalid source: {config['source']}")
        
    run_all_files(files, dir, config["source"], started=start_from_beginning)


if __name__ == "__main__":
    app()