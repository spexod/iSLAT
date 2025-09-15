import os
import sys

# Get the absolute path to the iSLAT directory
current_dir = os.path.dirname(os.path.abspath(__file__))
islat_dir = os.path.join(current_dir, "iSLAT")

# Add the current directory to Python path so we can import iSLAT
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Change to iSLAT directory for relative file paths to work
os.chdir(islat_dir)

from iSLAT.iSLATClass import iSLAT

if __name__ == "__main__":
    # Create an instance of the iSLAT class
    islat_instance = iSLAT()

    # Run the iSLAT application
    islat_instance.run()