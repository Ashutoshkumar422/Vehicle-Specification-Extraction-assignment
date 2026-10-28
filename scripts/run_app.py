import subprocess
import sys
from pathlib import Path

def main():
    """Launches the Streamlit Vehicle Specification Extraction web app."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print(f"Launching Streamlit application from: {app_path}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
