# Minecraft World to STL Converter

This Python script (mca_to_stl.py) allows you to convert a section of a Minecraft world into an STL file. You need to provide two sets of x, y, z coordinates to define the section, as well as the file path to the Minecraft world save folder.

## Prerequisites
- Python 3.x installed on your system.
- Required Python packages (numpy and numpy-stl) installed. You can install them via pip:
```pip install numpy numpy-stl ```

## Usage
1. Clone or download this repository to your local machine.
2. Navigate to the directory where mca_to_stl.py is located.
3. Open a terminal or command prompt in that directory.
4. Run the script with Python, providing the required parameters: ```python mca_to_stl.py <x1> <y1> <z1> <x2> <y2> <z2> <path_to_minecraft_save_folder>```
5. After providing the required information, the script will generate an STL file 'minectaft_model.stl' containing the selected region of the Minecraft world. The STL file will be saved in the current directory.

## Example
- First set of coordinates: x=0, y=64, z=0
- Second set of coordinates: x=15, y=79, z=15
- Minecraft world save folder path: C:\Users\username\AppData\Roaming\.minecraft\saves\my_world
- Full prompt: ```python mca_to_stl.py 0, 64, 0, 15, 79, 15, C:\Users\username\AppData\Roaming\.minecraft\saves\my_world```

The script will create an STL file containing the section of the Minecraft world defined by the coordinates (0, 64, 0) and (15, 79, 15), and save it in the current directory.

## Notes
- Ensure that you provide valid coordinates and the correct file path to the Minecraft world save folder.
- If you encounter import errors related to the stl package when running the script, ensure that you have only the numpy-stl package installed and not the stl package. You can uninstall the stl package using: ```pip uninstall stl```
