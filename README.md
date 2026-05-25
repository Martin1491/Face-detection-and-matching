CITS4402 Computer Vision Project - Face detection and matching
========================================================================

Team Members:
- [Zhi Wang]
- [Simona Han]
- [Bryan Zhang]

Description:
This project implements a fully contained face detection, skin color verification, 
similarity-based geometric alignment (125x125), and identity clustering pipeline with a Tkinter GUI.

Prerequisites & Installation:
1. Ensure Python 3.8+ is installed.
2. Install all required dependencies by running the following command in your terminal:
   pip install -r requirements.txt

How to Run the Script:
1. Open your terminal and navigate to this directory.
2. Execute the python script:
   python Wang_Han_Zhang_Script.py

GUI Instructions:
- Button A (Single Image): Click to select a single .jpg file. The aligned 125x125 face chips 
  with strict color-coded landmarks will overlay on the corners of the result frame.
- Button B (Bulk Processing): Click to select a folder containing sample images. 
  The system will dynamically cluster identities and natively export pure 125x125 face chips 
  into a directory named "Processed_Images" located at the SAME level as your selected folder.
