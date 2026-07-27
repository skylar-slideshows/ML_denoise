import subprocess
import sys
from pathlib import Path


def main():
	root = Path(__file__).resolve().parent
	scripts = [
		root / "src" / "1_preprocess.py",
		root / "src" / "2_encode.py",
		root / "src" / "3_make_split.py",
		root / "src" / "4_train.py",
		root / "src" / "5_infer.py",
	]

    #
	print(f"\n\n\n####### FULL DENOISING LSTM PIPELINE #######\nPlease ensure all training data and configs are in place.\n\n\n")

	for script in scripts:
		if not script.is_file():
			raise FileNotFoundError(f"Missing script {script}")

	for index, script in enumerate(scripts, start=1):
		print(f"\n[{index}/{len(scripts)}] \n\n\n####### STARTING {script.name}... #######\n\n\n")
		subprocess.run([sys.executable, str(script)], check=True, cwd=str(root))

	print("\nPipeline complete.")


if __name__ == "__main__":
	main()
