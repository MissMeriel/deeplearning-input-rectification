
C:\Users\Meriel\Documents\GitHub\deeplearning-input-rectification\venv-dqn\Scripts\python.exe C:/Users/Meriel/Documents/GitHub/deeplearning-input-rectification/simulation/take-a-lap-DDPGhuman-hardwarefix-writedata.py --transformation resdec --scenario straight --beamnginstance BeamNG.research --port 64156
ECHO "look for BeamNG keywd process"
wmic process get processid,parentprocessid,executablepath | find "BeamNG"
TASKKILL /IM BeamNG.research.x64.exe