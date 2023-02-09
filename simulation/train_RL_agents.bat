C:\Users\Meriel\Documents\GitHub\deeplearning-input-rectification\venv-dqn\Scripts\python.exe C:/Users/Meriel/Documents/GitHub/deeplearning-input-rectification/simulation/take-a-lap-DDPGhuman-hardwarefix-writedata.py --transformation resdec --scenario straight --beamnginstance BeamNG.research --port 64156
wmic process get processid,parentprocessid,executablepath | find "BeamNG"
TASKKILL /IM BeamNG.research.x64.exe
TIMEOUT 60
TASKKILL /IM BeamNG.research.x64.exe
TIMEOUT 60

C:\Users\Meriel\Documents\GitHub\deeplearning-input-rectification\venv-dqn\Scripts\python.exe C:/Users/Meriel/Documents/GitHub/deeplearning-input-rectification/simulation/take-a-lap-DDPGhuman-hardwarefix-writedata.py --transformation resdec --scenario winding --beamnginstance BeamNG.researchINSTANCE2 --port 64356
wmic process get processid,parentprocessid,executablepath | find "BeamNG"
TASKKILL /IM BeamNG.research.x64.exe
TIMEOUT 60
TASKKILL /IM BeamNG.research.x64.exe
TIMEOUT 60

C:\Users\Meriel\Documents\GitHub\deeplearning-input-rectification\venv-dqn\Scripts\python.exe C:/Users/Meriel/Documents/GitHub/deeplearning-input-rectification/simulation/take-a-lap-DDPGhuman-hardwarefix-writedata.py --transformation fisheye --scenario straight --beamnginstance BeamNG.researchINSTANCE3 --port 64556
wmic process get processid,parentprocessid,executablepath | find "BeamNG"
TASKKILL /IM BeamNG.research.x64.exe
TIMEOUT 60
TASKKILL /IM BeamNG.research.x64.exe
TIMEOUT 60

C:\Users\Meriel\Documents\GitHub\deeplearning-input-rectification\venv-dqn\Scripts\python.exe C:/Users/Meriel/Documents/GitHub/deeplearning-input-rectification/simulation/take-a-lap-DDPGhuman-hardwarefix-writedata.py --transformation fisheye --scenario winding --beamnginstance BeamNG.research --port 64756
wmic process get processid,parentprocessid,executablepath | find "BeamNG"
TASKKILL /IM BeamNG.research.x64.exe